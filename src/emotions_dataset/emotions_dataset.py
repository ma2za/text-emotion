import json
import os
import zipfile
from typing import List

import datasets
import pandas as pd
from datasets import ClassLabel, Value

_URLS = {
    "go_emotions": {
        "urls": [
            "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv",
            "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv",
            "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv",
        ],
        "license": "apache license 2.0"
    },
    "daily_dialog": {
        "urls": ["http://yanran.li/files/ijcnlp_dailydialog.zip"],
        "license": "CC BY-NC-SA 4.0"
    },
    "emotion": {
        "data": ["data/data.jsonl.gz"],
        "license": "educational/research"
    }
}

_CLASS_NAMES = [
    "no emotion",
    "happiness",
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


class EmotionsDatasetConfig(datasets.BuilderConfig):

    def __init__(self, features, label_classes, **kwargs):
        super().__init__(**kwargs)
        self.features = features
        self.label_classes = label_classes


class EmotionsDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        EmotionsDatasetConfig(
            name="all",
            label_classes=_CLASS_NAMES,
            features=["text", "label", "dataset", "license"]
        ),
        EmotionsDatasetConfig(
            name="go_emotions",
            label_classes=_CLASS_NAMES,
            features=["text", "label", "dataset", "license"]
        ),
        EmotionsDatasetConfig(
            name="daily_dialog",
            label_classes=_CLASS_NAMES,
            features=["text", "label", "dataset", "license"]
        ),
        EmotionsDatasetConfig(
            name="emotion",
            label_classes=_CLASS_NAMES,
            features=["text", "label", "dataset", "license"]
        )
    ]

    DEFAULT_CONFIG_NAME = "all"

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    'text': Value(dtype='string', id=None),
                    'label': ClassLabel(names=_CLASS_NAMES, id=None),
                    'dataset': Value(dtype='string', id=None),
                    'license': Value(dtype='string', id=None)
                }
            )
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        splits = []
        if self.config.name == "all":
            for k, v in _URLS.items():
                downloaded_files = dl_manager.download_and_extract(v.get("urls", v.get("data")))
                splits.append(datasets.SplitGenerator(name=k,
                                                      gen_kwargs={"filepaths": downloaded_files,
                                                                  "dataset": k,
                                                                  "license": v.get("license")}))
        else:
            k = self.config.name
            v = _URLS.get(k)
            downloaded_files = dl_manager.download_and_extract(v.get("urls", v.get("data")))
            splits.append(datasets.SplitGenerator(name=k,
                                                  gen_kwargs={"filepaths": downloaded_files,
                                                              "dataset": k,
                                                              "license": v.get("license")}))
        return splits

    def process_daily_dialog(self, filepaths, dataset):
        # TODO move outside
        emo_mapping = {0: "no emotion", 1: "anger", 2: "disgust",
                       3: "fear", 4: "happiness", 5: "sadness", 6: "surprise"}
        for i, filepath in enumerate(filepaths):
            if os.path.isdir(filepath):
                emotions = open(os.path.join(filepath, "ijcnlp_dailydialog/dialogues_emotion.txt"), "r").read()
                text = open(os.path.join(filepath, "ijcnlp_dailydialog/dialogues_text.txt"), "r").read()
            else:
                # TODO check if this can be removed
                archive = zipfile.ZipFile(filepath, 'r')
                emotions = archive.open("ijcnlp_dailydialog/dialogues_emotion.txt", "r").read().decode()
                text = archive.open("ijcnlp_dailydialog/dialogues_text.txt", "r").read().decode()
            emotions = emotions.split("\n")
            text = text.split("\n")

            for idx_out, (e, t) in enumerate(zip(emotions, text)):
                if len(t.strip()) > 0:
                    cast_emotions = [int(j) for j in e.strip().split(" ")]
                    cast_dialog = [d.strip() for d in t.split("__eou__") if len(d)]
                    for idx_in, (ce, ct) in enumerate(zip(cast_emotions, cast_dialog)):
                        uid = f"daily_dialog_{i}_{idx_out}_{idx_in}"
                        yield uid, {"text": ct,
                                    "id": uid,
                                    "dataset": dataset,
                                    "license": license,
                                    "label": emo_mapping[ce]}

    def _generate_examples(self, filepaths, dataset, license):
        if dataset == "go_emotions":
            for i, filepath in enumerate(filepaths):
                df = pd.read_csv(filepath)
                current_classes = list(set(df.columns).intersection(set(_CLASS_NAMES)))
                df = df[["text"] + current_classes]
                df = df[df[current_classes].sum(axis=1) == 1].reset_index(drop=True)
                for row_idx, row in df.iterrows():
                    uid = f"go_emotions_{i}_{row_idx}"
                    yield uid, {"text": row["text"],
                                "id": uid,
                                "dataset": dataset,
                                "license": license,
                                "label": row[current_classes][row == 1].index.item()}
        elif dataset == "daily_dialog":
            for d in self.process_daily_dialog(filepaths, dataset):
                yield d
        elif dataset == "emotion":
            emo_mapping = {0: "sadness", 1: "joy", 2: "love",
                           3: "anger", 4: "fear", 5: "surprise"}
            for i, filepath in enumerate(filepaths):
                with open(filepath, encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        uid = f"{dataset}_{idx}"
                        example = json.loads(line)
                        example.update({
                            "id": uid,
                            "dataset": dataset,
                            "license": license,
                            "label": emo_mapping[example["label"]]
                        })
                        yield uid, example
