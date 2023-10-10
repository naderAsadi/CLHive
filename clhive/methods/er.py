from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from . import register_method, BaseMethod
from ..data import ReplayBuffer
from ..utils import Logger


@register_method("er")
class ER(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        buffer: ReplayBuffer,
        n_replay_samples: int,
        logger: Logger = None,
        **kwargs,
    ) -> "ER":
        super().__init__(model=model, optimizer=optimizer, logger=logger)

        self.buffer = buffer
        self.n_replay_samples = n_replay_samples
        self.loss_func = nn.CrossEntropyLoss()

    @property
    def name(self) -> str:
        return "er"

    def process_inc(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.model(x=x, t=t)
        loss = self.loss_func(outputs.logits, y)

        return loss

    def process_re(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.model(x=x, t=t)
        loss = self.loss_func(outputs.logits, y)

        return loss

    def observe(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        inc_loss = self.process_inc(x, y, t)

        re_loss = 0
        if len(self.buffer) > 0:
            if self.n_replay_samples is None:
                self.n_replay_samples = x.size(0)

            re_data = self.buffer.sample(n_samples=self.n_replay_samples)
            re_loss = self.process_re(x=re_data["x"], y=re_data["y"], t=re_data["t"])

        loss = inc_loss + re_loss
        self.update(loss)

        self.buffer.add(batch={"x": x, "y": y, "t": t})

        return loss
