from typing import Any, Dict

import torch.nn as nn
from torchvision import transforms


class BaseTransform(nn.Module):
    """
    Class representing a data transform abstraction.
    """

    def __init__(self, mean=None, std=None):
        super().__init__()

        self.normalize = None
        if (mean is not None) and (std is not None):
            self.normalize = transforms.Normalize(mean, std)

        self.tfs = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, input):
        """
        The interface `__call__` is used to transform the input data. It should contain
        the actual implementation of data transform.

        Args:
            input: input image data
        """
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(**config)