import itertools
from typing import List

import pandas as pd
from datasets import Features, ClassLabel, Value, Dataset
from datasets import load_dataset, concatenate_datasets
from transformers import PreTrainedTokenizer


def daily_dialog_preprocessing():
    data = load_dataset("daily_dialog")
    data = concatenate_datasets([data["train"],
                                 data["validation"],
                                 data["test"]])

    def merge_data(examples):
        dd2emo = {
            1: "anger",
            3: "fear",
            4: "joy",
            5: "sadness",
            6: "surprise"
        }

        return {"chunks": [{"text": d, "label": dd2emo[e]} for d, e in zip(examples["dialog"], examples["emotion"]) if
                           e in [1, 3, 4, 5, 6] and len(d) < 85]}

    temp = data.map(merge_data, remove_columns=data.column_names)

    features = Features({'label': ClassLabel(names=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'], id=None),
                         'text': Value(dtype='string', id=None)})

    data = Dataset.from_pandas(pd.DataFrame(list(itertools.chain(*temp["chunks"]))),
                               features=features)
    return data


def go_emotions_preprocessing():
    data = load_dataset("go_emotions", "raw")

    data["train"] = data["train"].remove_columns(
        ['id', 'author', 'subreddit', 'link_id', 'parent_id', 'created_utc', 'rater_id', 'example_very_unclear',
         'admiration', 'amusement', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire',
         'disappointment',
         'disapproval', 'disgust', 'embarrassment', 'excitement', 'gratitude', 'grief', 'nervousness', 'optimism',
         'pride',
         'realization', 'relief', 'remorse', 'neutral'])

    features = Features({'label': ClassLabel(names=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'], id=None),
                         'text': Value(dtype='string', id=None)})

    def filter_data(*args):
        return sum(args) == 1

    data = data.filter(filter_data,
                       input_columns=['anger', 'fear', 'joy', 'love', 'sadness', 'surprise'])

    def map_data(examples):
        for k, v in examples.items():
            if v == 1:
                return {"label": k}
        raise Exception

    data = data.map(map_data,
                    remove_columns=['anger', 'fear', 'joy', 'love', 'sadness', 'surprise'])

    data = data["train"].cast_column("label", features["label"])

    return data


def load_emotion_datasets(datasets: List[str], tokenizer: PreTrainedTokenizer):
    dataset = load_dataset("emotion")

    if "daily_dialog" in datasets:
        df_support = daily_dialog_preprocessing()
        dataset["train"] = concatenate_datasets([dataset["train"], df_support])
    if "go_emotions" in datasets:
        df_support = go_emotions_preprocessing()
        dataset["train"] = concatenate_datasets([dataset["train"], df_support])

    def tokenization(sample):
        return tokenizer(sample["text"], padding=True, truncation=False)

    dataset = dataset.map(tokenization, batched=False, batch_size=None)

    dataset.set_format("torch", columns=["input_ids", "label"])

    return dataset
