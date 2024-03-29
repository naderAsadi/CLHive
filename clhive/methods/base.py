from cmath import log
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_table

from ..config import Config
from ..loggers import BaseLogger
from ..models import ContinualModel
from ..utils import get_optimizer


class BaseMethod(nn.Module):
    """[summary]"""

    def __init__(
        self,
        model: Union[ContinualModel, nn.Module],
        optim: torch.optim,
        logger: Optional[BaseLogger] = None,
        **kwargs,
    ) -> "BaseMethod":
        """_summary_

        Args:
            model (Union[ContinualModel, nn.Module]): _description_
            optim (torch.optim): _description_
            logger (Optional[BaseLogger], optional): _description_. Defaults to None.

        Returns:
            BaseMethod: _description_
        """

        super(BaseMethod, self).__init__()

        self.model = model
        self.optim = optim
        self.logger = logger

        self._model_history = {}
        self._current_task_id = 0

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def current_task(self) -> int:
        return self._current_task_id

    @property
    def one_sample_flop(self) -> Tuple[float, float]:
        """[summary]"""
        if not hasattr(self, "_train_cost"):
            input = torch.FloatTensor(size=(1,) + self.config.input_size).to(
                self.device
            )
            flops = FlopCountAnalysis(self.model, input)
            self._train_cost = flops.total() / 1e6  # MegaFlops
            self._train_flop_table = flop_count_table(flops)

        return self._train_cost, self._train_flop_table

    def get_model(self, task_id: int = None) -> Union[ContinualModel, nn.Module]:
        """[summary]

        Args:
            task_id (int, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if (task_id is None) or (task_id == self._current_task_id):
            return self.model

        assert (
            str(task_id) in self._model_history.keys()
        ), f"No trained model is available for task {task_id}."
        return self._model_history[str(task_id)]

    def set_model(self, model: Union[ContinualModel, nn.Module], task_id: int) -> None:
        """_summary_

        Args:
            model (Union[ContinualModel, nn.Module]): _description_
            task_id (int): _description_
        """
        assert model is not None, "Input model cannot be None."

        if (task_id is None) or (task_id == self._current_task_id):
            self.model = model
        else:
            self._model_history[str(task_id)] = model

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def update(self, loss: torch.FloatTensor) -> None:
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def observe(
        self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor
    ) -> torch.FloatTensor:
        raise NotImplementedError

    def predict(self, x: torch.FloatTensor, t: torch.FloatTensor) -> torch.FloatTensor:
        if t is None:
            t = 0
        return self.model(x, t)

    def on_task_start(self) -> None:
        """Callback executed at the start of each task."""
        pass

    def on_task_end(self) -> None:
        """Callback executed at the end of each task."""

        self._current_task_id += 1
