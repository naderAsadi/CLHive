from typing import Any, List, Optional, Tuple, Union

import torch

from torchcl.methods import register_method, BaseMethod
from torchcl.data.transforms import BaseTransform

@register_method("finetuning")
class FineTuning(BaseMethod):

    def __init__(self, config, model, logger, transform, optim) -> None:
        super().__init__(config, model, logger, transform, optim)

    @property
    def name(self):
        return "finetuning"

    def observe(self, data):
        aug_data = self.transform(data['x'])

        pred = self.model(aug_data)
        loss = self.loss(pred, data['y'])

        self.update(loss)
