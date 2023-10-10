from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from . import register_method
from .er import ER
from ..data import ReplayBuffer
from ..utils import Logger


@register_method("er_ace")
class ER_ACE(ER):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        buffer: ReplayBuffer,
        n_replay_samples: int,
        logger: Logger = None,
        **kwargs,
    ) -> "ER_ACE":
        super().__init__(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            logger=logger,
            n_replay_samples=n_replay_samples,
        )

        self.seen_so_far = []

    @property
    def name(self) -> str:
        return "er_ace"

    def process_inc(
        self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor
    ) -> torch.FloatTensor:
        """get loss from incoming data"""

        present = y.unique()
        self.seen_so_far = list(set(self.seen_so_far + present.tolist()))

        # process data
        outputs = self.model(x=x, t=t)
        logits = outputs.logits
        mask = torch.zeros_like(logits)

        # unmask current classes
        mask[:, present] = 1

        # unmask unseen classes
        mask[:, max(self.seen_so_far) :] = 1

        if t[0].item() > 0:
            logits = logits.masked_fill(mask == 0, -1e9)

        loss = self.loss(logits, y)

        return loss
