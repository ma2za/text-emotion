import itertools
import os
import random
from functools import partial
from typing import Dict

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import ClassLabel, Features, Value
from datasets import concatenate_datasets
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding, RobertaTokenizerFast

from configuration_roberta_emotion import RobertaEmotionConfig
from modeling_roberta_emotion import RobertaEmotion

load_dotenv()

wandb.login(key=os.getenv("KEY"))

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

"""## Tokenizer"""

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")


def tokenization(sample):
    return tokenizer(sample["text"], padding=True, truncation=False)


"""## Dataset"""

dataset = load_dataset("emotion")

daily_dialog = load_dataset("daily_dialog")

daily_dialog = concatenate_datasets([daily_dialog["train"], daily_dialog["validation"], daily_dialog["test"]])

go_emotions = load_dataset("go_emotions", "raw")

go_emotions["train"] = go_emotions["train"].remove_columns(
    ['id', 'author', 'subreddit', 'link_id', 'parent_id', 'created_utc', 'rater_id', 'example_very_unclear',
     'admiration', 'amusement', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
     'disapproval', 'disgust', 'embarrassment', 'excitement', 'gratitude', 'grief', 'nervousness', 'optimism', 'pride',
     'realization', 'relief', 'remorse', 'neutral'])

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


def filter_go_emotions(*args):
    return sum(args) == 1


go_emotions = go_emotions.filter(filter_go_emotions,
                                 input_columns=['anger', 'fear', 'joy', 'love', 'sadness', 'surprise'])


def map_go_emotions(examples):
    for k, v in examples.items():
        if v == 1:
            return {"label": k}
    raise Exception


go_emotions = go_emotions.map(map_go_emotions,
                              remove_columns=['anger', 'fear', 'joy', 'love', 'sadness', 'surprise'])


def merge_daily_dialog(examples):
    dd2emo = {
        1: "anger",
        3: "fear",
        4: "joy",
        5: "sadness",
        6: "surprise"
    }

    return {"chunks": [{"text": d, "label": dd2emo[e]} for d, e in zip(examples["dialog"], examples["emotion"]) if
                       e in [1, 3, 4, 5, 6] and len(d) < 85]}


temp = daily_dialog.map(merge_daily_dialog, remove_columns=daily_dialog.column_names)

features = Features({'label': ClassLabel(names=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'], id=None),
                     'text': Value(dtype='string', id=None)})
go_emotions = go_emotions["train"].cast_column("label", features["label"])
daily_dialog = Dataset.from_pandas(pd.DataFrame(list(itertools.chain(*temp["chunks"]))),
                                   features=features)

# DATASETS = ["emotion"]
DATASETS = ["emotion", "daily_dialog", "go_emotions"]

if len(DATASETS) != 1:
    dataset["train"] = concatenate_datasets([dataset["train"], daily_dialog, go_emotions])

dataset = dataset.map(tokenization, batched=False, batch_size=None)

dataset.set_format("torch", columns=["input_ids", "label"])

train_dataset = dataset["train"]
train_dataset.remove_columns(["text"])

valid_dataset = dataset["validation"]
valid_dataset.remove_columns(["text"])

"""## Model"""

RobertaEmotionConfig.register_for_auto_class()

RobertaEmotion.register_for_auto_class("AutoModel")

model_config = {
    "id2label": id2label,
    "label2id": label2id,
    "hidden_size": 768,
    "num_labels": 6,
    "position_embedding_type": "relative_key_query",
    "hidden_act": "gelu_new",
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
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            acc, f1 = compute_metrics(outputs.logits.argmax(-1).detach().cpu(),
                                      batch["labels"].detach().cpu())
            total_acc += acc * len(batch["labels"])
            total_f1 += f1 * len(batch["labels"])
            total_samples += len(batch["labels"])
            total_loss += outputs.loss.detach().cpu() * len(batch["labels"])
    return total_acc / total_samples, total_f1 / total_samples, total_loss / total_samples


def train(model, optimizer, train_loader, valid_loader, config, lr_scheduler=None, epochs=0):
    best_f1 = 0

    unfreeze_patience = 5
    current_patience = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = torch.tensor(0.0).to(device)
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            model.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            epoch_loss += outputs.loss.detach() * len(inputs["input_ids"])

            if lr_scheduler is not None:
                lr_scheduler.step()
        valid_acc, valid_f1, valid_loss = evaluation(model, valid_loader)

        if best_f1 < valid_f1:
            best_f1 = valid_f1
            torch.save(model.state_dict(), "pytorch_model.bin")
            current_patience = 0
        else:
            current_patience += 1
        if current_patience >= unfreeze_patience and lr_scheduler is None:
            break


data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="longest",
    return_tensors="pt",
)


def train_roberta(config: Dict):
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        collate_fn=data_collator,
        drop_last=False,
        num_workers=0,
        shuffle=True,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=4,
        collate_fn=data_collator,
        drop_last=False,
        num_workers=0,
        pin_memory=True
    )

    num_training_steps = float(len(train_loader) * config["epochs"])

    num_warmup_steps = num_training_steps * config["warmup_ratio"]

    model = RobertaEmotion(emotion_config).to(device)
    best_model = wandb.restore('pytorch_model.bin', run_path='meraxes/emotion_classifier/9ajzjglr')
    temp = torch.load("pytorch_model.bin", map_location=torch.device('cpu'))
    model.load_state_dict(temp)

    valid_acc, valid_f1, valid_loss = evaluation(model, valid_loader)

    model.backbone.requires_grad_(False)

    for n, p in model.backbone.encoder.named_parameters():
        if "distance_embedding" in n:
            torch.nn.init.xavier_normal_(p)
            p.requires_grad_(True)

    torch.nn.init.xavier_normal_(model.classifier[1].weight)

    optimizer = AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.999),
                      eps=1e-08, weight_decay=0.01)

    train(model, optimizer, train_loader, valid_loader, config, epochs=10000)

    print("FIRST STEP DONE!")

    optimizer = AdamW(model.parameters(), lr=config["lr"] * 0.001, betas=(0.9, 0.999),
                      eps=1e-08, weight_decay=0.01)

    def lr_lambda(x: float, warmup: float, total: float):
        return x / warmup if x < warmup else (total - x) / (total - warmup)

    lr_scheduler = LambdaLR(optimizer, partial(lr_lambda, warmup=num_warmup_steps,
                                               total=num_training_steps))

    model.requires_grad_(True)

    train(model, optimizer, train_loader, valid_loader, config, lr_scheduler, config["epochs"])

    return model


train_config = {"batch_size": 128,
                "epochs": 100,
                "lr": 1e-02,
                "warmup_ratio": 0.2,
                "datasets": DATASETS}

model = train_roberta(train_config)