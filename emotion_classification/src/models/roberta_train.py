import itertools
import os
import random
from functools import partial
from typing import Dict

import numpy as np
import pandas as pd
import torch
from datasets import ClassLabel, Features, Value
from datasets import Dataset
from datasets import concatenate_datasets
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import RobertaTokenizer
from transformers.data.data_collator import default_data_collator

from emotion_classification.src.roberta_emotion.configuration_roberta_emotion import RobertaEmotionConfig
from emotion_classification.src.roberta_emotion.modeling_roberta_emotion import RobertaEmotion

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

"""## Tokenizer"""

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def tokenization(sample):
    return tokenizer(sample["text"], padding=True, truncation=True)


"""## Dataset"""

dataset = load_dataset("emotion")

daily_dialog = load_dataset("daily_dialog")

daily_dialog = concatenate_datasets([daily_dialog["train"], daily_dialog["validation"], daily_dialog["test"]])


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

daily_dialog = Dataset.from_pandas(pd.DataFrame(list(itertools.chain(*temp["chunks"]))),
                                   features=features)

DATASETS = ["emotion"]
# DATASETS = ["emotion", "daily_dialog"]

if len(DATASETS) != 1:
    dataset["train"] = concatenate_datasets([dataset["train"], daily_dialog])

dataset = dataset.map(tokenization, batched=True, batch_size=None)

dataset.set_format("torch", columns=["input_ids", "label"])

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
    "vocab_size": 50265,
    "max_position_embeddings": 514,
    "type_vocab_size": 1
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


def train(model, checkpoint_dir, optimizer, lr_scheduler, train_loader,
          valid_loader, tune_flag=False, config={}):
    best_f1 = 0
    for epoch in range(config["epochs"]):
        model.train()
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            model.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)

        valid_acc, valid_f1, valid_loss = evaluation(model, valid_loader)

        if best_f1 < valid_f1:
            best_f1 = valid_f1
            path = os.path.join(checkpoint_dir, "pytorch_model.bin")
            torch.save(model.state_dict(), path)


def train_roberta(config: Dict, checkpoint_dir: str = None):
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        collate_fn=default_data_collator,
        drop_last=False,
        num_workers=0,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=64,
        collate_fn=default_data_collator,
        drop_last=False,
        num_workers=0,
        pin_memory=True
    )

    num_training_steps = float(len(train_loader) * config["epochs"])

    num_warmup_steps = num_training_steps * config["warmup_ratio"]

    model = RobertaEmotion(emotion_config).to(device)
    optimizer = AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.999), eps=1e-08)

    def lr_lambda(x: float, warmup: float, total: float):
        return x / warmup if x < warmup else (total - x) / (total - warmup)

    lr_scheduler = LambdaLR(optimizer, partial(lr_lambda, warmup=num_warmup_steps,
                                               total=num_training_steps))
    train(model, checkpoint_dir, optimizer, lr_scheduler, train_loader, valid_loader, config=config)
    return model


train_config = {"batch_size": 128,
                "epochs": 25,
                "lr": 1e-05,
                "warmup_ratio": 0.2,
                "datasets": DATASETS}

model = train_roberta(train_config, checkpoint_dir=".")
