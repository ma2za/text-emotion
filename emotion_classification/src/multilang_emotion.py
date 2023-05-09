import torch
from easynmt import EasyNMT
from transformers import AutoModel, AutoTokenizer, AutoConfig


def emotion(text: str) -> str:
    """

    :param text:
    :return:
    """

    translator = EasyNMT("opus-mt")

    en_text = translator.translate(text, target_lang="en")

    config = AutoConfig.from_pretrained("ma2za/roberta-emotion", trust_remote_code=True)

    model = AutoModel.from_pretrained("ma2za/roberta-emotion", trust_remote_code=True, config=config)

    tokenizer = AutoTokenizer.from_pretrained("ma2za/roberta-emotion", trust_remote_code=True)

    with torch.no_grad():
        output = model(tokenizer.encode(en_text, return_tensors="pt"))
    return model.config.id2label[output.logits.argmax(-1).cpu().detach().numpy()[0]]
