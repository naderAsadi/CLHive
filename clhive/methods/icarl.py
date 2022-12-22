from typing import Any, List, Optional, Tuple, Union
import copy
import torch
import torch.nn.functional as F

from . import register_method
from .er import ER
from ..data import ReplayBuffer
from ..loggers import BaseLogger
from ..models import ContinualModel


@register_method("icarl")
class ICARL(ER):
    def __init__(
        self,
        model: Union[ContinualModel, torch.nn.Module],
        optim: torch.optim,
        buffer: ReplayBuffer,
        logger: Optional[BaseLogger] = None,
        n_replay_samples: Optional[int] = None,
        **kwargs,
    ) -> "ICARL":
        """_summary_

        Args:
            model (Union[ContinualModel, torch.nn.Module]): _description_
            optim (torch.optim): _description_
            buffer (ReplayBuffer): _description_
            logger (Optional[BaseLogger], optional): _description_. Defaults to None.
            n_replay_samples (Optional[int], optional): _description_. Defaults to None.

        Returns:
            ICARL: _description_
        """
        super().__init__(
            model=model,
            optim=optim,
            buffer=buffer,
            logger=logger,
            n_replay_samples=n_replay_samples,
        )

        self.D_C = 1
        self._centroids = None
        self._old_model = None

        self.loss = torch.nn.BCEWithLogitsLoss(reduction="sum")

    @property
    def name(self) -> str:
        return "icarl"

    @torch.no_grad()
    def _calculate_centroids(self) -> None:
        buffer = self.buffer
        n_batches = len(buffer) // 512 + 1

        hid_size = self.model.forward_backbone(buffer.bx[:2]).size(-1)

        arr_D = torch.arange(hid_size).to(buffer.bx.device)

        protos = buffer.bx.new_zeros(size=(self.args.n_classes, hid_size))
        count = buffer.by.new_zeros(size=(self.args.n_classes,))

        for i in range(n_batches):
            idx = range(i * 512, min(len(buffer), (i + 1) * 512))
            xx, yy = buffer.bx[idx], buffer.by[idx]

            hid_x = self.model.forward_backbone(xx)

            b_proto = torch.zeros_like(protos)
            b_count = torch.zeros_like(count)

            b_count.scatter_add_(0, yy, torch.ones_like(yy))

            out_idx = arr_D.view(1, -1) + yy.view(-1, 1) * hid_size
            b_proto = (
                b_proto.view(-1)
                .scatter_add(0, out_idx.view(-1), hid_x.view(-1))
                .view_as(b_proto)
            )

            protos += b_proto
            count += b_count

        self._centroids = protos / count.view(-1, 1)

        # mask out unobserved centroids
        self._centroids[count < 1] = -1e9

    def process_inc(
        self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor
    ) -> torch.FloatTensor:
        """_summary_

        Args:
            x (torch.FloatTensor): _description_
            y (torch.FloatTensor): _description_
            t (torch.FloatTensor): _description_

        Returns:
            torch.FloatTensor: _description_
        """
        logits = self.model(x, t)
        label = F.one_hot(y, num_classes=logits.size(-1)).float()

        loss = self.loss(logits.view(-1), label.view(-1)).sum()
        loss = loss / logits.size(0)

        return loss

    def process_re(
        self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor
    ) -> torch.FloatTensor:
        """_summary_

        Args:
            x (torch.FloatTensor): _description_
            y (torch.FloatTensor): _description_
            t (torch.FloatTensor): _description_

        Returns:
            torch.FloatTensor: _description_
        """
        loss = 0

        if self._old_model is not None:
            with torch.no_grad():
                tgt = F.sigmoid(self._old_model(x, t))

            logits = self.model(x, t)
            loss = self.loss(logits.view(-1), tgt.view(-1)) / logits.size(0)

        return loss

    def observe(
        self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor
    ) -> torch.FloatTensor:
        """_summary_

        Args:
            x (torch.FloatTensor): _description_
            y (torch.FloatTensor): _description_
            t (torch.FloatTensor): _description_

        Returns:
            torch.FloatTensor: _description_
        """
        loss = super().observe(x, y, t)

        # mask centroids as out of sync
        self._centroids = None

        return loss

    def on_task_end(self) -> None:
        self._old_model = copy.deepcopy(self.model)
        self._old_model.eval()
        super().on_task_end()
