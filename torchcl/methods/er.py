from typing import Any, List, Optional, Tuple, Union

import torch

from torchcl.methods import register_method, BaseMethod
from torchcl.data import Buffer
from torchcl.data.transforms import BaseTransform


@register_method("er")
class ER(BaseMethod):
    def __init__(self, config, model, logger, transform, optim) -> None:
        super().__init__(config, model, logger, transform, optim)

        self.buffer = Buffer(device, capacity, input_size)
        self.loss = torch.nn.CrossEntropyLoss()

    @property
    def name(self):
        return 'er'

    

    def process_inc(self, data):
        aug_data = self.transform(data['x'])

        pred = self.model(aug_data, data['t'])
        loss = self.loss(pred, data['y'])

        return loss

    def observe(self, data):
        pass