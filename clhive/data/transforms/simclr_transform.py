from typing import Any, Dict

import torch
from torchvision import transforms

from . import register_transform
from .base_transform import BaseTransform


@register_transform("simclr")
class SimCLRTransform(BaseTransform):
    def __init__(self, size, mean=None, std=None):
        super(SimCLRTransform, self).__init__(mean, std)

        self.tfs = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, input):
        """
        The interface `__call__` is used to transform the input data. It should contain
        the actual implementation of data transform.

        Args:
            input: input image data
        """

        return self.tfs(input)

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(**config)
