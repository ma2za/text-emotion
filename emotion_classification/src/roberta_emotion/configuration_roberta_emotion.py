from transformers import PretrainedConfig


class RobertaEmotionConfig(PretrainedConfig):
    model_type = "ma2za/roberta-emotion"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
