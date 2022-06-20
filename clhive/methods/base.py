from cmath import log
import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_table

from ..models import ModelWrapper
from ..data.transforms import BaseTransform
from ..utils import get_optimizer


class BaseMethod(nn.Module):
    """[summary]

    """

    def __init__(
        self,
        config,
        model : ModelWrapper,
        logger,
        transform: BaseTransform = BaseTransform,
        optim: torch.optim = None,
    ) -> None:

        super(BaseMethod, self).__init__()

        self.model = model
        self.logger = logger
        self.transform = transform
        self.config = config

        if optim is None:
            optim = get_optimizer(self.config)
        self.optim = optim

        self._model_history = {}
        self._current_task_id = 0

    
    @property
    def name(self):
        raise NotImplementedError

    @property
    def current_task(self):
        return self._current_task_id

    @property
    def one_sample_flop(self):
        """[summary]
        """
        if not hasattr(self, '_train_cost'):
            input = torch.FloatTensor(size=(1,) + self.config.input_size).to(self.device)
            flops = FlopCountAnalysis(self.model, input)
            self._train_cost = flops.total() / 1e6 # MegaFlops
            self._train_flop_table = flop_count_table(flops)

        return self._train_cost, self._train_flop_table
    
    def get_model(self, task_id: int = None):
        """[summary]

        Args:
            task_id (int, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if (task_id is None) or (task_id == self._current_task_id):
            return self.model

        assert str(task_id) in self._model_history.keys(), (
            f"No trained model is available for task {task_id}."
        )
        return self._model_history[str(task_id)]

    def set_model(self, model: ModelWrapper, task_id: int):
        """[summary]

        Args:
            model (ModelWrapper): [description]
            task_id (int): [description]
        """
        assert model is not None, (
            "Input model cannot be None."
        )

        if (task_id is None) or (task_id == self._current_task_id):
            self.model = model
        else:
            self._model_history[str(task_id)] = model

    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
    
    def update(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def observe(self, data: Dict[str, torch.Tensor]):
        raise NotImplementedError

    def predict(self, data, task_id) -> torch.Tensor:
        if self.config.eval.scenario == 'single_head':
            task_id = 0
        return self.model(data, task_id)

    def on_task_start(self):
        pass

    def on_task_end(self):
        """[summary]
        """
        # save snapshots and other stuff

        self._current_task_id = min(self._current_task_id + 1, self.config.data.n_tasks - 1)



    
