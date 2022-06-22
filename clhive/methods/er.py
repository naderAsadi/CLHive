from typing import Any, List, Optional, Tuple, Union

import torch

from . import register_method, BaseMethod
from ..data import Buffer
from ..models import ContinualModel


@register_method("er")
class ER(BaseMethod):
    def __init__(
        self,
        model: Union[ContinualModel, torch.nn.Module],
        optim: torch.optim,
        logger=None,
    ) -> None:
        super().__init__(config, model, logger, transform, optim)

        device = torch.device("cuda")
        self.buffer = Buffer(
            device, self.config.data.mem_size, self.config.data.image_size
        )
        self.loss = torch.nn.CrossEntropyLoss()

    @property
    def name(self):
        return "er"

    def process(self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor):
        pred = self.model(x, t)
        loss = self.loss(pred, y)

        return loss

    def observe(self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor):
        inc_loss = self.process(x, y, t)

        re_loss = 0
        if len(self.buffer) > 0:
            re_data = self.buffer.sample({"amt": 20, "exclude_task": None,})
            re_loss = self.process(*re_data)

        self.update(inc_loss + re_loss)
        self.buffer.add(data)

        return inc_loss + re_loss
