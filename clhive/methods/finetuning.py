from typing import Any, List, Optional, Tuple, Union
import torch

from . import register_method, BaseMethod
from ..models import ContinualModel


@register_method("finetuning")
class FineTuning(BaseMethod):
    def __init__(
        self,
        model: Union[ContinualModel, torch.nn.Module],
        optim: torch.optim,
        logger=None,
    ) -> None:
        super().__init__(model, optim, logger)

        self.loss = torch.nn.CrossEntropyLoss()

    @property
    def name(self):
        return "finetuning"

    def observe(self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor):
        pred = self.model(x, t)
        loss = self.loss(pred, y)

        self.update(loss)

        return loss
