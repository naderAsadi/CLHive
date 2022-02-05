from cmath import log
import copy
from statistics import mode
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis as FCA

from torchcl.models import ModelWrapper
from torchcl.data.transforms import BaseTransform


class BaseMethod(nn.Module):
    """[summary]

    """

    def __init__(
        self,
        model : ModelWrapper,
        logger,
        transform: BaseTransform,
        config
    ) -> None:

        super(BaseMethod, self).__init__()

        self.model = model
        self.logger = logger
        self.transform = transform
        self.config = config

    
    @property
    def name(self):
        raise NotImplementedError

    @property
    def cost(self):
        raise NotImplementedError

    @property
    def one_sample_flop(self):
        """[summary]
        """
        if not hasattr(self, '_train_cost'):
            input = torch.FloatTensor(size=(1,) + self.config.input_size).to(self.device)
            flops = FCA(self.model, input)
            self._train_cost = flops.total() / 1e6 # MegaFlops

        return self._train_cost
