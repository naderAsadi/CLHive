from typing import Any, List, Optional, Tuple, Union
import torch

from . import register_method, BaseMethod
from ..data import ReplayBuffer
from ..loggers import BaseLogger
from ..models import ContinualModel


@register_method("er")
class ER(BaseMethod):
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
        super().__init__(model, optim, logger)

        self.buffer = buffer
        self.n_replay_samples = n_replay_samples
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

            if self.n_replay_samples is None:
                self.n_replay_samples = x.size(0)

            re_data = self.buffer.sample(n_samples=self.n_replay_samples)
            re_loss = self.process(x=re_data["x"], y=re_data["y"], t=re_data["t"])

        loss = inc_loss + re_loss
        self.update(loss)

        self.buffer.add(batch={"x": x, "y": y, "t": t})

        return loss
