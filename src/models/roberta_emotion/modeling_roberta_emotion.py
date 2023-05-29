from collections import OrderedDict
from operator import attrgetter

import torch
from torch import Tensor
from transformers import PreTrainedModel, RobertaModel
from transformers import RobertaConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from configuration_roberta_emotion import RobertaEmotionConfig


class LoRA(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, weight: Tensor,
                 bias: Tensor, alpha: int, r: int):
        super().__init__(in_features, out_features, bias is not None)
        self.delta_weight = torch.nn.Sequential(OrderedDict([
            ("A", torch.nn.Linear(in_features, 8, bias=False)),
            ("B", torch.nn.Linear(8, out_features, bias=False))
        ]))

        self.alpha = alpha
        self.r = r

        torch.nn.init.zeros_(self.delta_weight.B.weight)
        torch.nn.init.normal_(self.delta_weight.A.weight)

        self.weight = weight
        self.bias = bias

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input) + self.delta_weight(input) * (self.alpha / self.r)


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

        if len(kwargs.get("lora", [])) != 0:
            for name, module in self.backbone.named_modules():
                if any([i in name for i in kwargs.get("lora")]):
                    module_name, attr_name = name.rsplit(".", 1)
                    module: torch.nn.Module = attrgetter(module_name)(self.backbone)
                    attr: torch.nn.Linear = attrgetter(name)(self.backbone)
                    module.__setattr__(attr_name, LoRA(
                        in_features=attr.in_features,
                        out_features=attr.out_features,
                        weight=attr.weight,
                        bias=attr.bias,
                        alpha=8,
                        r=8
                    ))

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
