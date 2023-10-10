import torch
import torch.nn as nn

from . import register_method, BaseMethod
from ..utils import Logger


@register_method("finetuning")
class FineTuning(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        logger: Logger = None,
        **kwargs,
    ) -> "FineTuning":
        super().__init__(model=model, optimizer=optimizer, logger=logger)

        self.loss_func = nn.CrossEntropyLoss()

    @property
    def name(self):
        return "finetuning"

    def observe(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        outputs = self.model(x, t)
        loss = self.loss_func(outputs.logits, y)

        self.update(loss)

        return loss
