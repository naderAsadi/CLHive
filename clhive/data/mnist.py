from typing import Any, Callable, Dict, Optional, Tuple, Union
from PIL import Image

from torchvision.datasets.mnist import MNIST

from . import register_dataset
from .continual_dataset import ContinualDataset


@register_dataset("seqmnist")
class SeqMNIST(ContinualDataset):

    _IMAGE_SIZE = 28

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        normalize_targets_per_task: Optional[bool] = False,
        train: Optional[bool] = True,
        download: Optional[bool] = True,
    ) -> None:

        dataset = MNIST(root, train=train, download=download)

        super().__init__(
            dataset=dataset,
            transform=transform,
            normalize_targets_per_task=normalize_targets_per_task,
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
        return cls(root=root, transform=transform, train=train, download=True)
