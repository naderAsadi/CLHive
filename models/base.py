from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for models in torchCL.

    A model refers either to a specific architecture (e.g. ResNet50) or a
    family of architectures (e.g. ResNet). Models can take arguments in the
    constructor in order to configure different behavior (e.g.
    hyperparameters).  torchCL models must implement :func:`from_config` in
    order to allow instantiation from a configuration file. Like regular
    PyTorch models, models must also implement :func:`forward`, where
    the bulk of the inference logic lives.

    Models also have some advanced functionality for production
    fine-tuning systems. For example, we allow users to train a trunk
    model and then attach heads to the model via the attachable
    blocks.  Making your model support the trunk-heads paradigm is
    completely optional.
    """

    def __init__(self):
        super().__init__()
        self._heads = nn.ModuleDict()
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseModel":
        """Instantiates a Model from a configuration.

        Args:
            config (Dict): A configuration for the Model.

        Returns:
            A torch.nn.Module instance.
        """

        raise NotImplementedError
    
    def forward(self, x):
        """
        Perform computation of blocks in the order define in get_blocks.
        """

        raise NotImplementedError