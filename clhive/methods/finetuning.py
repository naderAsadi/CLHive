from typing import Any, List, Optional, Tuple, Union

import torch

from . import register_method, BaseMethod


@register_method("finetuning")
class FineTuning(BaseMethod):
    def __init__(self, config, model, logger, transform, optim) -> None:
        super().__init__(config, model, logger, transform, optim)

        self.loss = torch.nn.CrossEntropyLoss()

    @property
    def name(self):
        return "finetuning"

    def observe(self, data):
        aug_data = self.transform(data["x"])

        pred = self.model(aug_data, data["t"])
        loss = self.loss(pred, data["y"])

        self.update(loss)

        return loss
