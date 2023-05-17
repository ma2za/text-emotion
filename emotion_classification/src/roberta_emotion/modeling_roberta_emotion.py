import torch
from transformers import PreTrainedModel, RobertaModel, RobertaConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from .configuration_roberta_emotion import RobertaEmotionConfig


class RobertaEmotion(PreTrainedModel):
    config_class = RobertaEmotionConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        roberta_base_config = RobertaConfig.from_pretrained("roberta-base",
                                                            **config.to_dict(),
                                                            num_labels=config.num_labels)

        self.backbone = RobertaModel.from_pretrained("roberta-base", False, config=roberta_base_config)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, int(config.hidden_size / 2)),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(int(config.hidden_size / 2), config.num_labels)
        )

    def forward(self, input_ids, labels=None, attention_mask=None):
        logits = self.classifier(self.backbone(input_ids).last_hidden_state[:, 0, :])

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)
