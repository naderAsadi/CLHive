import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_method
from .er import ER
from ..data import ReplayBuffer
from ..utils import Logger


@register_method("icarl")
class ICARL(ER):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        buffer: ReplayBuffer,
        n_replay_samples: int,
        num_classes: int,
        hidden_dim: int,
        logger: Logger = None,
        **kwargs,
    ) -> "ICARL":
        super().__init__(
            model=model,
            optim=optim,
            buffer=buffer,
            logger=logger,
            n_replay_samples=n_replay_samples,
        )

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.D_C = 1
        self._centroids = None
        self._old_model = None

        self.loss_func = nn.BCEWithLogitsLoss(reduction="sum")

    @property
    def name(self) -> str:
        return "icarl"

    @torch.no_grad()
    def _calculate_centroids(self) -> None:
        buffer = self.buffer
        n_batches = len(buffer) // 512 + 1

        arr_D = torch.arange(self.hidden_dim).to(buffer.device)

        protos = buffer.bx.new_zeros(size=(self.num_classes, self.hidden_dim))
        count = buffer.by.new_zeros(size=(self.num_classes,))

        for i in range(n_batches):
            idx = range(i * 512, min(len(buffer), (i + 1) * 512))
            xx, yy = buffer.bx[idx], buffer.by[idx]

            outputs = self.model(xx)
            hid_x = outputs.hidden_states

            b_proto = torch.zeros_like(protos)
            b_count = torch.zeros_like(count)

            b_count.scatter_add_(0, yy, torch.ones_like(yy))

            out_idx = arr_D.view(1, -1) + yy.view(-1, 1) * self.hidden_dim
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
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.model(x, t)
        logits = outputs.logits
        label = F.one_hot(y, num_classes=logits.size(-1)).float()

        loss = self.loss_func(logits.view(-1), label.view(-1)).sum()
        loss = loss / logits.size(0)

        return loss

    def process_re(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        loss = 0

        if self._old_model is not None:
            with torch.no_grad():
                tgt = F.sigmoid(self._old_model(x, t))

            logits = self.model(x, t).logits
            loss = self.loss_func(logits.view(-1), tgt.view(-1)) / logits.size(0)

        return loss

    def observe(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        loss = super().observe(x, y, t)

        # mask centroids as out of sync
        self._centroids = None

        return loss

    def on_task_end(self) -> None:
        self._old_model = copy.deepcopy(self.model)
        self._old_model.eval()
        super().on_task_end()
