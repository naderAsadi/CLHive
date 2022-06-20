from typing import Any, Callable, Dict, Optional, Tuple, Union
from PIL import Image

from torchvision.datasets.mnist import MNIST

from . import register_dataset
from .base import BaseDataset
from ..transforms import get_transform, BaseTransform


@register_dataset("seq_mnist")
class MNISTDataset(BaseDataset):

    _IMAGE_SIZE = 28

    def __init__(
        self,
        root: str, 
        transform: Optional[Union[BaseTransform, Callable]],
        train: bool,
        download: bool = True
    ) -> None:

        dataset = MNIST(root, train=train, download=download)
        if transform is None:
            transform = BaseTransform()
        
        super().__init__(
            dataset=dataset, transform=transform
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any], train: bool) -> "MNISTDataset":
        """Instantiates a MNISTDataset from a configuration.
        Args:
            config: A configuration for a MNISTDataset.
                See :func:`__init__` for parameters expected in the config.
        Returns:
            A MNISTDataset instance.
        """
        
        root = config.get("root")
        train = train
        transform_config = config.get("transform")
        download = config.get("download")

        transform = get_transform(transform_config)
        return cls(
            root=root, 
            transform=transform,
            train=train,
            download=True
        )