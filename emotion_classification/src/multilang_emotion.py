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

    translator = EasyNMT("opus-mt")
    src = _language_detection(text)

    # TODO group languages
    sentences = []

    tokenizer = AutoTokenizer.from_pretrained("ma2za/roberta-emotion", trust_remote_code=True)

    for source_lang, sentence in zip(src, text):
        en_text = translator.translate(sentence, source_lang=source_lang, target_lang="en")
        sentences.append(tokenizer.encode(en_text, return_tensors="pt"))

    config = AutoConfig.from_pretrained("ma2za/roberta-emotion", trust_remote_code=True)

    model = AutoModel.from_pretrained("ma2za/roberta-emotion", trust_remote_code=True, config=config)
    
    # TODO batch prediction
    output = []
    with torch.no_grad():
        for sentence in sentences:
            prediction = model.config.id2label[model(sentence).logits.argmax(-1).cpu().detach().numpy()[0]]
            output.append(prediction)
    return output
