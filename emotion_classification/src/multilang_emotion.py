from typing import List, Union

import fasttext
import torch
from easynmt import EasyNMT
from transformers import AutoModel, AutoTokenizer, AutoConfig


def _language_detection(text: List[str]) -> List[str]:
    """

    :param text:
    :return:
    """

    pretrained_lang_model = "../data/lid.176.bin"
    try:
        lang_model = fasttext.load_model(pretrained_lang_model)
    except ValueError:
        raise Exception("The fasttext language detection model is not present!")
    src = lang_model.predict(text, k=1)
    src = [lang[0].replace("__label__", "") for lang in src[0]]
    return src


def emotion(text: Union[str, List[str]]) -> List[str]:
    """

    :param text:
    :return:
    """

    if isinstance(text, str):
        text = [text]

    src = _language_detection(text)

    translator = EasyNMT("opus-mt")

    # TODO optimize grouping
    inputs = {}
    for src_lang, sentence in zip(src, text):
        sentence_list = inputs.get(src_lang, [])
        sentence_list.append(translator.translate(sentence, source_lang=src_lang, target_lang="en"))
        inputs[src_lang] = sentence_list

    tokenizer = AutoTokenizer.from_pretrained("ma2za/roberta-emotion", trust_remote_code=True)

    config = AutoConfig.from_pretrained("ma2za/roberta-emotion", trust_remote_code=True)

    model = AutoModel.from_pretrained("ma2za/roberta-emotion", trust_remote_code=True, config=config)

    output = []
    with torch.no_grad():
        for src_lang, sentences in inputs.items():
            # TODO break long sentences
            input_ids = tokenizer(sentences, padding=True, truncation=False,
                                  return_attention_mask=False, return_tensors="pt").get("input_ids")
            prediction = model(input_ids).logits.argmax(-1).cpu().detach().numpy()
            prediction = [model.config.id2label[x] for x in prediction]
            output.extend(prediction)

    return output
