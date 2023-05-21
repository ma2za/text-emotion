import os
import random
from collections import Counter
from functools import partial
from typing import Dict

import datasets
import numpy as np
import torch
import wandb
from datasets import load_dataset
from evaluate import evaluator
from sklearn.metrics import accuracy_score, f1_score
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm.notebook import tqdm
from transformers import DataCollatorWithPadding, RobertaTokenizer

from configuration_roberta_emotion import RobertaEmotionConfig
from modeling_roberta_emotion import RobertaEmotion

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Commented out IPython magic to ensure Python compatibility.
# %env WANDB_PROJECT=emotion_classifier

wandb.login(key="dd76be81cdc7cbf86e3fd3ab08c05d73a6cb815d")

device = "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = ["emotion", "daily_dialog", "go_emotions"]

train_config = {"batch_size": 128,
                "epochs": 15,
                "lr": 1e-04,
                "warmup_ratio": 0.2,
                "pooling": True,
                "pretrain": False,
                "balance": True,
                "smoothing": 0.1,
                "datasets": DATASETS}

"""## Dataset"""

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

id2label = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

label2id = {
    "sadness": 0,
    "joy": 1,
    "love": 2,
    "anger": 3,
    "fear": 4,
    "surprise": 5
}

support_dataset = load_dataset("ma2za/emotions_dataset")

dataset = load_dataset("dair-ai/emotion")

labels = dataset["train"].features["label"]

labels_mapping = {}
for i, e in enumerate(support_dataset["go_emotions"].features["label"].names):
    if e in labels.names:
        labels_mapping[i] = labels.str2int(e)

support_dataset = support_dataset.filter(lambda x: x["label"] in labels_mapping.keys() and len(x["text"]) < 100)

support_dataset = support_dataset.map(lambda x: {"label": labels_mapping[x["label"]]})

support_dataset = support_dataset.cast_column("label", dataset["train"].features["label"])

support_dataset = support_dataset.remove_columns(['id', 'dataset', 'license'])

dataset["train"] = datasets.concatenate_datasets([dataset["train"],
                                                  support_dataset["go_emotions"],
                                                  support_dataset["daily_dialog"]])


def tokenization(sample):
    return tokenizer(sample["text"], padding=True, truncation=False)


dataset = dataset.map(tokenization, batched=False, batch_size=None,
                      remove_columns=["text"])

dataset.set_format("torch", columns=["input_ids", "label"])

if train_config["balance"]:
    counts = dict(Counter(dataset["train"]["label"].numpy()))
    weights = [(1 / 6) / counts[w.item()] for w in dataset["train"]["label"]]
    sampler = WeightedRandomSampler(weights, len(weights))

"""## Model"""

RobertaEmotionConfig.register_for_auto_class()

RobertaEmotion.register_for_auto_class("AutoModel")

model_config = {
    "id2label": id2label,
    "label2id": label2id,
    "hidden_size": 768,
    "num_labels": 6,
    "position_embedding_type": "absolute",
    "hidden_act": "gelu",
    "vocab_size": 50265,
    "max_position_embeddings": 514,
    "type_vocab_size": 1,
}

emotion_config = RobertaEmotionConfig(**model_config)

"""## Training"""


def compute_metrics(preds, labels):
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return acc, f1


def evaluation(model, dataloader):
    model.eval()
    total_samples, total_loss, total_acc, total_f1 = 0, 0, 0, 0
    all_preds = torch.tensor([], dtype=torch.int64)
    all_labels = torch.tensor([], dtype=torch.int64)
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            preds = outputs.logits.argmax(-1).detach().cpu()
            labels = batch["labels"].detach().cpu()
            acc, f1 = compute_metrics(preds, labels)
            total_acc += acc * len(batch["labels"])
            total_f1 += f1 * len(batch["labels"])
            total_samples += len(batch["labels"])
            total_loss += outputs.loss.detach().cpu() * len(batch["labels"])
            all_preds = torch.concat((all_preds, preds))
            all_labels = torch.concat((all_labels, labels))
    return total_acc / total_samples, total_f1 / total_samples, total_loss / total_samples, list(
        all_preds.numpy()), list(all_labels.numpy())


def train(model, optimizer, train_loader, valid_loader, config, lr_scheduler=None, epochs=0):
    best_f1 = 0

    unfreeze_patience = 5
    current_patience = 0

    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        epoch_loss = torch.tensor(0.0).to(device)
        epoch_samples = 0
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            model.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(**inputs)
            epoch_loss += outputs.loss.detach() * len(inputs["input_ids"])
            epoch_samples += len(inputs["input_ids"])

            scaler.scale(outputs.loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if lr_scheduler is not None:
                lr_scheduler.step()
        valid_acc, valid_f1, valid_loss, predictions, labels = evaluation(model, valid_loader)

        if best_f1 < valid_f1:
            best_f1 = valid_f1
            torch.save(model.state_dict(), "pytorch_model.bin")
            wandb.save("pytorch_model.bin")
            current_patience = 0
        else:
            current_patience += 1

        wandb.log({"eval/loss": valid_loss, "eval/f1": valid_f1,
                   "eval/accuracy": valid_acc,
                   "train/patience": current_patience,
                   "train/lr": float(optimizer.param_groups[0]['lr']),
                   "train/loss": epoch_loss.cpu().item() / epoch_samples,
                   "conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                           y_true=labels, preds=predictions,
                                                           class_names=list(label2id.keys()))
                   })

        if current_patience >= unfreeze_patience and lr_scheduler is None:
            break


data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="longest",
    return_tensors="pt",
    max_length=20
)

train_loader = DataLoader(
    dataset["train"],
    batch_size=4,
    collate_fn=data_collator,
    drop_last=False,
    num_workers=4,
    sampler=sampler,
    pin_memory=True
)

for batch in train_loader:
    print(batch)


def pretrain(model, train_loader, valid_loader, config):
    model.backbone.requires_grad_(False)
    model.classifier.requires_grad_(True)
    if model.pooling:
        model.backbone.pooler.requires_grad_(True)
    else:
        model.custom_pooling.requires_grad_(True)

    for n, p in model.backbone.encoder.named_parameters():
        if "distance_embedding" in n:
            torch.nn.init.xavier_normal_(p)
            p.requires_grad_(True)

    torch.nn.init.xavier_normal_(model.classifier[-1].weight)

    optimizer = AdamW(model.parameters(), lr=config["lr"] * 1000, betas=(0.9, 0.999),
                      eps=1e-08, weight_decay=0.01)

    train(model, optimizer, train_loader, valid_loader, config, epochs=10000)
    return model


def train_roberta(config: Dict):
    train_loader = DataLoader(
        dataset["train"],
        batch_size=int(config["batch_size"]),
        collate_fn=data_collator,
        drop_last=False,
        num_workers=4,
        sampler=sampler if config.get("balance") else None,
        pin_memory=True
    )

    valid_loader = DataLoader(
        dataset["validation"],
        batch_size=64,
        collate_fn=data_collator,
        drop_last=False,
        num_workers=0,
        pin_memory=True
    )

    num_training_steps = float(len(train_loader) * config["epochs"])

    num_warmup_steps = num_training_steps * config["warmup_ratio"]

    model = RobertaEmotion(emotion_config, pooling=config["pooling"]).to(device)

    wandb.init(project="emotion_classifier", config={**model_config, **config})

    if config["pretrain"]:
        pretrain(model, train_loader, valid_loader, config)

    optimizer = AdamW(model.parameters(),
                      lr=config["lr"],
                      betas=(0.9, 0.999),
                      eps=1e-08,
                      weight_decay=0.01)

    def lr_lambda(x: float, warmup: float, total: float):
        return x / warmup if x < warmup else (total - x) / (total - warmup)

    lr_scheduler = LambdaLR(optimizer, partial(lr_lambda,
                                               warmup=num_warmup_steps,
                                               total=num_training_steps))

    model.requires_grad_(True)

    train(model, optimizer, train_loader, valid_loader,
          config, lr_scheduler, config["epochs"])

    wandb.finish()
    return model


model = train_roberta(train_config)

best_model = wandb.restore('pytorch_model.bin', run_path="meraxes/emotion_classifier/pnm8mm6l")

model = RobertaEmotion(emotion_config).to(device)

model_state = torch.load(os.path.join(".", "pytorch_model.bin"))
model.load_state_dict(model_state)

model_state = torch.load(os.path.join(".", "pytorch_model.bin"))
model.load_state_dict(model_state)

model.push_to_hub("roberta-emotion")
tokenizer.push_to_hub("roberta-emotion")

"""## Evaluation"""

task_evaluator = evaluator("text-classification")

results = task_evaluator.compute(
    model_or_pipeline=model,
    tokenizer=tokenizer,
    data="emotion",
    subset="split",
    split="test",
    metric="accuracy",
    label_mapping=label2id,
    strategy="bootstrap",
    n_resamples=10,
    random_state=0
)

results
