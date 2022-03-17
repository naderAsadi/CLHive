from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class BaseHead(nn.Module):
    """Base class for heads in torchCL.

    A head is a regular :class:`torch.nn.Module` that can be attached to a
    pretrained model. This enables a form of transfer learning: utilizing a
    model trained for one dataset to extract features that can be used for
    other problems. A head must be attached to a :class:`models.ClassyBlock`
    within a :class:`models.ClassyModel`.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseHead":
        """Instantiates a Head from a configuration.

        Args:
            config (Dict): A configuration for the Head.

        Returns:
            A BaseHead(torch.nn.Module) instance.
        """

        raise NotImplementedError
    
    def forward(self, x):
        """
        Performs inference on the head.

        This is a regular PyTorch method, refer to :class:`torch.nn.Module` for
        more details
        """

        raise NotImplementedError