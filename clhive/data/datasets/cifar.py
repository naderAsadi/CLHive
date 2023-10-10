from typing import Any, Callable, Dict, Optional, Sequence, Union
from torchvision import transforms as T
from torchvision.datasets.cifar import CIFAR10, CIFAR100

from . import register_dataset
from .base import BaseDataset


@register_dataset("cifar10")
class CIFAR10Dataset(BaseDataset, CIFAR10):
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        root: str,
        transform: Callable,
        train: bool = True,
        download: bool = True,
    ) -> None:
        super().__init__(
            root,
            train=train,
            transform=transform,
            download=download,
        )

    @classmethod
    def from_config(
        cls,
        data_config: Dict[str, Any],
        split: str = "train",
        **kwargs,
    ):
        if split == "train":
            transform = T.Compose(
                [
                    T.RandomCrop(data_config.image_size, padding=4),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.3),
                    T.ToTensor(),
                    T.Normalize(mean=CIFAR10Dataset._MEAN, std=CIFAR10Dataset._STD),
                ]
            )
        elif split == "test":
            transform = T.Compose(
                [
                    T.Resize(data_config.image_size),
                    T.ToTensor(),
                    T.Normalize(mean=CIFAR10Dataset._MEAN, std=CIFAR10Dataset._STD),
                ]
            )
        else:
            raise ValueError(
                f"`split` should be in [`train`, `test`], but {split} is entered."
            )

        return cls(
            root=data_config.root,
            transform=transform,
            train=True if split == "train" else False,
            download=True,
        )


@register_dataset("cifar100")
class CIFAR100Dataset(BaseDataset, CIFAR100):
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        root: str,
        transform: Callable,
        train: bool = True,
        download: bool = True,
    ) -> None:
        super().__init__(
            root,
            train=train,
            download=download,
            transform=transform,
        )

    @classmethod
    def from_config(
        cls,
        data_config: Dict[str, Any],
        split: str = "train",
        **kwargs,
    ):
        if split == "train":
            transform = T.Compose(
                [
                    T.RandomCrop(data_config.image_size, padding=4),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.3),
                    T.ToTensor(),
                    T.Normalize(mean=CIFAR100Dataset._MEAN, std=CIFAR100Dataset._STD),
                ]
            )
        elif split == "test":
            transform = T.Compose(
                [
                    T.Resize(data_config.image_size),
                    T.ToTensor(),
                    T.Normalize(mean=CIFAR100Dataset._MEAN, std=CIFAR100Dataset._STD),
                ]
            )
        else:
            raise ValueError(
                f"`split` should be in [`train`, `test`], but {split} is entered."
            )

        return cls(
            root=data_config.root,
            transform=transform,
            train=True if split == "train" else False,
            download=True,
        )
