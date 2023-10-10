from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_method
from .er import ER
from ..data import ReplayBuffer
from ..utils import Logger


@register_method("der")
class DER(ER):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        buffer: ReplayBuffer,
        n_replay_samples: int,
        logger: Logger = None,
        **kwargs,
    ) -> "DER":
        super().__init__(
            model=model,
            optim=optim,
            buffer=buffer,
            logger=logger,
            n_replay_samples=n_replay_samples,
        )

        self.alpha = 1

    @property
    def name(self) -> str:
        return "der"

    def process_inc(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        # process data
        outputs = self.model(x=x, t=t)
        loss = self.loss(outputs.logits, y)

        return loss, outputs.logits.detach()

    def process_re(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        re_logits: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(x=x, t=t)
        loss = self.alpha * F.mse_loss(outputs.logits, re_logits)

        return loss

    def observe(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        inc_loss, logits = self.process_inc(x, y, t)

        re_loss = 0
        if len(self.buffer) > 0:
            if self.n_replay_samples is None:
                self.n_replay_samples = x.size(0)

            re_data = self.buffer.sample(n_samples=self.n_replay_samples)
            re_loss = self.process_re(
                x=re_data["x"],
                y=re_data["y"],
                t=re_data["t"],
                re_logits=re_data["logits"],
            )

        loss = inc_loss + re_loss
        self.update(loss)

        self.buffer.add(batch={"x": x, "y": y, "t": t, "logits": logits})

        return loss


@register_method("derpp")
class DERpp(DER):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        buffer: ReplayBuffer,
        n_replay_samples: int,
        logger: Logger = None,
        **kwargs,
    ) -> "DERpp":
        super().__init__(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            n_replay_samples=n_replay_samples,
            logger=logger,
        )

        self.beta = 1

    @property
    def name(self) -> str:
        return "der++"

    def process_re(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        re_logits: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(x=x, t=t)
        o1, o2 = outputs.logits.chunk(2)

        x1, x2 = x.chunk(2)
        y1, y2 = y.chunk(2)
        l1, l2 = re_logits.chunk(2)

        aa = F.mse_loss(o1, l1)
        bb = self.loss(o2, y2)

        loss = self.alpha * aa + self.beta * bb

        return loss
