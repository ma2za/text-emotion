import os
from typing import List, Union

import fasttext
import requests
import torch
from easynmt import EasyNMT
from transformers import AutoModel, AutoTokenizer, AutoConfig

__all__ = ["Detector"]

CONFIG = {
    "model": {"name": "ma2za/roberta-emotion", "lang": ["en"]}
}

DEFAULT_TRANSLATE_CACHE = os.path.expanduser("~/.cache/text-emotion")

if not os.path.isdir(DEFAULT_TRANSLATE_CACHE):
    os.makedirs(DEFAULT_TRANSLATE_CACHE, exist_ok=True)


class Detector:

    def __init__(self, emotion_language: str = "en", batch_size: int = 4):

        self.emotion_language = emotion_language
        self.translator = EasyNMT("opus-mt")
        self.batch_size = batch_size

        # TODO check cache models, device_map, int8
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG.get("model", {}).get("name"), trust_remote_code=True)

        config = AutoConfig.from_pretrained(CONFIG.get("model", {}).get("name"), trust_remote_code=True)

        self.model = AutoModel.from_pretrained(CONFIG.get("model", {}).get("name"), trust_remote_code=True,
                                               config=config)

    @staticmethod
    def __language_detection(text: List[str]) -> List[str]:
        """

        :param text:
        :return:
        """

        fasttext_path = os.path.join(DEFAULT_TRANSLATE_CACHE, "fasttext")
        if not os.path.isdir(fasttext_path):
            os.makedirs(fasttext_path, exist_ok=True)

        fasttext_model = os.path.join(fasttext_path, "lid.176.bin")
        if not os.path.exists(fasttext_model):
            resp = requests.get("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
            with open(fasttext_model, "wb") as f:
                f.write(resp.content)

        try:
            lang_model = fasttext.load_model(fasttext_model)
        except ValueError:
            raise Exception("The fasttext language detection model is not present!")
        text = [t.replace("\n", " ") for t in text]
        src = lang_model.predict(text, k=1)
        src = [lang[0].replace("__label__", "") for lang in src[0]]
        return src

    def detect(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """

        :param text:
        :return:
        """

        return_list = True
        if isinstance(text, str):
            text = [text]
            return_list = False

        src = Detector.__language_detection(text)

        # TODO optimize grouping
        grouped_inputs = {}
        for src_lang, sentence in zip(src, text):
            sentence_list = grouped_inputs.get(src_lang, [])
            sentence_list.append(sentence)
            grouped_inputs[src_lang] = sentence_list

        model_langs = CONFIG.get("model", {}).get("lang", [])
        inputs = []
        for src_lang, sentences in grouped_inputs.items():
            if src_lang not in model_langs:
                inputs.extend(self.translator.translate(sentences, source_lang=src_lang, target_lang="en"))
            else:
                inputs.extend(sentences)

        output = []
        self.model.eval()
        with torch.no_grad():
            for i in range(int(len(inputs) / self.batch_size) + 1):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                if i >= len(inputs):
                    break
                # TODO break long sentences
                input_ids = self.tokenizer(inputs[start:end], padding=True, truncation=True,
                                           return_attention_mask=False, return_tensors="pt").get("input_ids")
                prediction = self.model(input_ids).logits.argmax(-1).cpu().detach().numpy()
                prediction = [self.model.config.id2label[x] for x in prediction]
                output.extend(prediction)
        # TODO consider labels in source language
        labels = self.translator.translate(output, source_lang="en", target_lang=self.emotion_language)
        return labels if return_list else labels[0]
