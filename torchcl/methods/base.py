from cmath import log
import copy
from statistics import mode
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_table

from torchcl.models import ModelWrapper
from torchcl.data.transforms import BaseTransform
from torchcl.utils import get_optimizer


class BaseMethod(nn.Module):
    """[summary]

    """

    def __init__(
        self,
        model : ModelWrapper,
        logger,
        config,
        transform: Optional[BaseTransform] = BaseTransform,
        optim: Optional[torch.optim] = None,
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

    
    @property
    def name(self):
        raise NotImplementedError

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
        if task_id is None:
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
        self._model_history[str(task_id)] = model

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError

    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
    
    def update(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def observe(self, data):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def on_task_start(self):
        raise NotImplementedError

    def on_task_end(self):
        raise NotImplementedError

    
