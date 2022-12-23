from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from . import register_method
from .er import ER
from ..data import ReplayBuffer
from ..loggers import BaseLogger
from ..models import ContinualModel


def store_grad(params, grads, grad_dims):
    """
    This stores parameter gradients of past tasks.
    pp: parameters
    grads: gradients
    grad_dims: list with number of parameters per layers
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

    @property
    def name(self) -> str:
        return "agem"
