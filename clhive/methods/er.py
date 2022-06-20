from typing import Any, List, Optional, Tuple, Union

import torch

from . import register_method, BaseMethod
from ..data import Buffer
from ..data.transforms import BaseTransform


@register_method("er")
class ER(BaseMethod):
    def __init__(self, config, model, logger, transform, optim) -> None:
        super().__init__(config, model, logger, transform, optim)

        device = torch.device("cuda")
        print(device)
        self.buffer = Buffer(
            device, self.config.data.mem_size, self.config.data.image_size
        )
        self.loss = torch.nn.CrossEntropyLoss()

    @property
    def name(self):
        return "er"

    def process_re(self, data):
        print(data)
        aug_data = self.transform(data["x"])

        pred = self.model(aug_data, data["t"])
        loss = self.loss(pred, data["y"])

        return loss

    def process_inc(self, data):
        aug_data = self.transform(data["x"])

        pred = self.model(aug_data, data["t"])
        loss = self.loss(pred, data["y"])

        return loss

    def observe(self, data):

        inc_loss = self.process_inc(data)

        re_loss = 0
        if len(self.buffer) > 0:
            re_data = self.buffer.sample({"amt": 20, "exclude_task": None,})
            re_loss = self.process_re(re_data)

        print(data["x"].device)
        self.update(inc_loss + re_loss)
        self.buffer.add(data)

        return inc_loss + re_loss
