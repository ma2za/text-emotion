import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .configuration_roberta_emotion import RobertaEmotionConfig


class RobertaEmotion(PreTrainedModel):
    config_class = RobertaEmotionConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.backbone = AutoModel.from_pretrained("roberta-base", config)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(config.hidden_size, config.num_labels)
        )
        torch.nn.init.xavier_normal_(self.classifier[1].weight)

    def forward(self, input_ids, labels=None, attention_mask=None):
        logits = self.classifier(self.backbone(input_ids).last_hidden_state[:, 0, :])

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)
