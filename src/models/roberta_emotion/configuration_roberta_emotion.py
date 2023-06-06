from transformers import RobertaConfig


class RobertaEmotionConfig(RobertaConfig):
    model_type = "ma2za/roberta-text_emotion"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
