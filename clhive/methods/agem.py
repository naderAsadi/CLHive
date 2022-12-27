from typing import Any, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from . import register_method
from .er import ER
from ..data import ReplayBuffer
from ..loggers import BaseLogger
from ..models import ContinualModel


def store_grad(params, grads, grad_dims):
    """This stores parameter gradients of past tasks.
    params: parameters.

    Args:
        params (_type_): parameters
        grads (_type_): gradients
        grad_dims (_type_): list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[: count + 1])
            grads[begin:end].copy_(param.grad.data.view(-1))
        count += 1


def overwrite_grad(params, new_grad, grad_dims):
    """This is used to overwrite the gradients with a new gradient
    vector, whenever violations occur.

    Args:
        params (_type_): parameters
        new_grad (_type_): corrected gradient
        grad_dims (_type_): list storing number of parameters at each layer
    """
    cnt = 0
    for param in params():
        param.grad = torch.zeros_like(param.data)
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[: cnt + 1])
        this_grad = new_grad[beg:en].contiguous().view(param.data.size())
        param.grad.data.copy_(this_grad)
        cnt += 1


def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger


@register_method("agem")
class AGEM(ER):
    def __init__(
        self,
        model: Union[ContinualModel, torch.nn.Module],
        optim: torch.optim,
        buffer: ReplayBuffer,
        logger: Optional[BaseLogger] = None,
        n_replay_samples: Optional[int] = None,
        **kwargs,
    ) -> "AGEM":
        """_summary_

        Args:
            model (Union[ContinualModel, torch.nn.Module]): _description_
            optim (torch.optim): _description_
            buffer (ReplayBuffer): _description_
            logger (Optional[BaseLogger], optional): _description_. Defaults to None.
            n_replay_samples (Optional[int], optional): _description_. Defaults to None.

        Returns:
            AGEM: _description_
        """
        super().__init__(
            model=model,
            optim=optim,
            buffer=buffer,
            logger=logger,
            n_replay_samples=n_replay_samples,
        )

        self.grad_dims = []
        for param in model.parameters():
            self.grad_dims += [param.data.numel()]

        device = next(self.model.parameters()).device
        self.grad_inc = torch.zeros(np.sum(self.grad_dims)).to(device)
        self.grad_re = torch.zeros(np.sum(self.grad_dims)).to(device)

    @property
    def name(self) -> str:
        return "agem"

    def overwrite_grad(self, projected_grad: torch.Tensor) -> None:
        overwrite_grad(self.model.parameters, projected_grad, self.grad_dims)

    def process_re(
        self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor
    ) -> torch.FloatTensor:
        # store grad
        store_grad(self.model.parameters, self.grad_inc, self.grad_dims)

        # clear grad buffers
        self.model.zero_grad()

        # rehearsal grad
        pred = self.model(x, t)
        re_loss = self.loss(pred, y)
        re_loss.backward()
        store_grad(self.model.parameters, self.grad_re, self.grad_dims)

        # potentially project incoming gradient
        dot_p = torch.dot(self.grad_inc, self.grad_re)
        if dot_p < 0.0:
            proj_grad = project(gxy=self.grad_inc, ger=self.grad_re)
        else:
            proj_grad = self.grad_inc

        self.overwrite_grad(proj_grad)

        return re_loss

    def observe(
        self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor
    ) -> torch.FloatTensor:

        self.optim.zero_grad()

        # --- training --- #
        inc_loss = self.process_inc(x=x, y=y, t=t)
        inc_loss.backward()

        re_loss, re_data = 0.0, None
        if len(self.buffer) > 0:

            # -- rehearsal starts ASAP. No task id is used
            if self._current_task_id > 0:
                re_data = self.buffer.sample(n_samples=self.n_replay_samples)
                re_loss = self.process_re(
                    x=re_data["x"], y=re_data["y"], t=re_data["t"]
                )

        self.optim.step()

        # --- buffer overhead --- #
        self.buffer.add(batch={"x": x, "y": y, "t": t})

        return inc_loss + re_loss


@register_method("agempp")
class AGEMpp(AGEM):
    @property
    def name(self) -> str:
        return "agempp"

    def overwrite_grad(self, projected_grad):
        overwrite_grad(
            self.model.parameters, projected_grad + self.grad_re, self.grad_dims
        )
