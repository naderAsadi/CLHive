from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from ..utils import Logger


class BaseMethod(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        logger: Logger = None,
        **kwargs,
    ) -> "BaseMethod":
        super(BaseMethod, self).__init__()

        self.model = model
        self.optimizer = optimizer
        if logger is None:
            logger = Logger()
        self.logger = logger

        self._model_history = {}
        self._current_task_id = 0

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def current_task(self) -> int:
        return self._current_task_id

    def get_model(self, task_id: int) -> nn.Module:
        if task_id == self._current_task_id:
            return self.model

        assert (
            str(task_id) in self._model_history.keys()
        ), f"No trained model is available for task {task_id}. Avaiable models: {list(self._model_history.keys())}"
        return self._model_history[str(task_id)]

    def set_model(self, model: nn.Module, task_id: int) -> None:
        assert model is not None, "Input model cannot be None."

        if task_id == self._current_task_id:
            self.model = model
        else:
            self._model_history[str(task_id)] = model

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def update(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def observe(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        return self.model(x=x, t=t)

    def on_task_start(self) -> None:
        """Callback executed at the start of each task."""
        pass

    def on_task_end(self) -> None:
        """Callback executed at the end of each task."""

        self._current_task_id += 1
