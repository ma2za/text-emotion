import importlib
import os
from typing import List, Union

import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig

__all__ = ["Detector"]

CONFIG = {
    "model": {"name": "ma2za/roberta-emotion", "lang": ["en"]}
}

DEFAULT_TRANSLATE_CACHE = os.path.expanduser("~/.cache/text-emotion")

if not os.path.isdir(DEFAULT_TRANSLATE_CACHE):
    os.makedirs(DEFAULT_TRANSLATE_CACHE, exist_ok=True)


def is_translate_available():
    return importlib.util.find_spec("fasttext") is not None and importlib.util.find_spec("easynmt") is not None


class Detector:

    def __init__(self, emotion_language: str = "en", batch_size: int = 4):

        self.emotion_language = emotion_language
        self.batch_size = batch_size

        # TODO check cache models, device_map, int8
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG.get("model", {}).get("name"), trust_remote_code=True)

        config = AutoConfig.from_pretrained(CONFIG.get("model", {}).get("name"), trust_remote_code=True)

        self.model = AutoModel.from_pretrained(CONFIG.get("model", {}).get("name"), trust_remote_code=True,
                                               config=config)

        if is_translate_available():
            from easynmt import EasyNMT

            self.translator = EasyNMT("opus-mt")

    @staticmethod
    def __language_detection(text: List[str]) -> List[str]:
        """

        Detect source languages for the list of input sentences.

        :param text:
        :return:
        """

        import fasttext
        import requests

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

    def __translate_text(self, text: List[str]) -> List[str]:
        """

        Translate input text to english only if the detect source language is not supported by
        the emotion model.

        :param text:
        :return:
        """
        src = Detector.__language_detection(text)

        grouped_inputs = {}
        for i, (src_lang, sentence) in enumerate(zip(src, text)):
            sentence_dict = grouped_inputs.get(src_lang)
            if sentence_dict is None:
                sentence_dict = {}
                grouped_inputs[src_lang] = sentence_dict
            sentence_dict[i] = sentence

        model_langs = CONFIG.get("model", {}).get("lang", [])
        inputs = []
        for src_lang, sentences in grouped_inputs.items():
            input_sentences = list(sentences.values())
            if src_lang not in model_langs:
                inputs.extend(
                    list(zip(list(sentences.keys()),
                             self.translator.translate(input_sentences, source_lang=src_lang, target_lang="en"))))
            else:
                inputs.extend(list(sentences.items()))
        # TODO that's ridiculous
        return list(dict(sorted(inputs, key=lambda x: x[0])).keys())

    def detect(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """

        :param text:
        :return:
        """

        return_list = True
        if isinstance(text, str):
            text = [text]
            return_list = False

        inputs = self.__translate_text(text) if is_translate_available() else text

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
