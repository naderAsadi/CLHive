from typing import Callable, Tuple
from PIL import Image
from torch import Tensor
from torchvision.datasets.mnist import MNIST

from . import register_dataset
from .base import BaseDataset
from ..config import DataConfig


@register_dataset("mnist")
class MNISTDataset(BaseDataset, MNIST):
    def __init__(
        self,
        root: str,
        transform: Callable,
        target_transform: Callable = None,
        train: bool = True,
        download: bool = True,
    ) -> None:
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    @classmethod
    def from_config(
        cls,
        data_config: DataConfig,
        split: str = "train",
        **kwargs,
    ):
        if split == "train":
            transform = T.Compose(
                [
                    T.RandomCrop(data_config.image_size, padding=4),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor(),
                ]
            )
        else:
            transform = T.Compose(
                [
                    T.Resize(data_config.image_size),
                    T.ToTensor(),
                ]
            )

        return cls(
            root=data_config.root,
            transform=transform,
            train=True if split == "train" else False,
            download=True,
        )

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, int]:
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.task_id
