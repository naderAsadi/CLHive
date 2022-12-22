from typing import Any, List, Optional, Tuple, Union
import torch

from . import register_method
from .er import ER
from ..data import ReplayBuffer
from ..loggers import BaseLogger
from ..models import ContinualModel


@register_method("er_ace")
class ER_ACE(ER):
    def __init__(
        self,
        model: Union[ContinualModel, torch.nn.Module],
        optim: torch.optim,
        buffer: ReplayBuffer,
        logger: Optional[BaseLogger] = None,
        n_replay_samples: Optional[int] = None,
        **kwargs,
    ) -> "ER":
        """_summary_

        Args:
            model (Union[ContinualModel, torch.nn.Module]): _description_
            optim (torch.optim): _description_
            buffer (ReplayBuffer): _description_
            logger (Optional[BaseLogger], optional): _description_. Defaults to None.
            n_replay_samples (Optional[int], optional): _description_. Defaults to None.

        Returns:
            ER: _description_
        """
        super().__init__(
            model=model,
            optim=optim,
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
        logits = self.model(x, t)
        mask = torch.zeros_like(logits)

        # unmask current classes
        mask[:, present] = 1

        # unmask unseen classes
        mask[:, max(self.seen_so_far) :] = 1

        if t[0].item() > 0:
            logits = logits.masked_fill(mask == 0, -1e9)

        loss = self.loss(logits, y)

        return loss
