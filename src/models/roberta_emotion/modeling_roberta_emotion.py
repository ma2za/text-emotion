import torch
from transformers import PreTrainedModel, RobertaModel, RobertaConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from .configuration_roberta_emotion import RobertaEmotionConfig


class RobertaEmotion(PreTrainedModel):
    config_class = RobertaEmotionConfig

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.smoothing = kwargs.get("smoothing", 0.0)
        roberta_base_config = RobertaConfig.from_pretrained("roberta-base",
                                                            **config.to_dict(),
                                                            num_labels=config.num_labels)

        self.backbone = RobertaModel.from_pretrained("roberta-base",
                                                     kwargs.get("pooling", True),
                                                     config=roberta_base_config)
        self.custom_pooling = None
        if not kwargs.get("pooling", True):
            self.custom_pooling = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.hidden_size),
                torch.nn.GELU(),
            )

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(config.hidden_size, config.num_labels)
        )

    def forward(self, input_ids, labels=None, attention_mask=None):
        hidden = self.backbone(input_ids).last_hidden_state[:, 0, :]
        if self.custom_pooling is not None:
            hidden = self.custom_pooling(hidden)
        logits = self.classifier(hidden)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=self.smoothing)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)
