from typing import Any, Callable, Dict, Optional, Sequence, Union

from torchvision.datasets.cifar import CIFAR10, CIFAR100

from . import register_dataset
from .continual_dataset import ContinualDataset


class CIFARDataset(ContinualDataset):

    _CIFAR_TYPE = None
    _DEFAULT_N_TASKS = None
    _MEAN = (0.4914, 0.4822, 0.4465)
    _STD = (0.2023, 0.1994, 0.2010)
    _IMAGE_SIZE = 32

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        normalize_targets_per_task: Optional[bool] = False,
        train: Optional[bool] = True,
        download: Optional[bool] = True,
    ) -> None:

        assert self._CIFAR_TYPE in [
            "cifar10",
            "cifar100",
        ], "CIFARDataset must be subclassed and a valid _CIFAR_TYPE provided"

        if self._CIFAR_TYPE == "cifar10":
            dataset = CIFAR10(root, train=train, download=download)
        if self._CIFAR_TYPE == "cifar100":
            dataset = CIFAR100(root, train=train, download=download)

        if transform is None:
            transform = BaseTransform(mean=CIFARDataset._MEAN, std=CIFARDataset._STD)

        super().__init__(
            dataset=dataset,
            transform=transform,
            normalize_targets_per_task=normalize_targets_per_task,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any], train: bool) -> "CIFARDataset":
        """Instantiates a CIFARDataset from a configuration.
        Args:
            config: A configuration for a CIFARDataset.
                See :func:`__init__` for parameters expected in the config.
        Returns:
            A CIFARDataset instance.
        """

        root = config.get("root")
        train = train
        transform_config = config.get("transform")
        download = config.get("download")

        transform = get_transform(transform_config)
        return cls(root=root, transform=transform, train=train, download=True)


@register_dataset("cifar10")
class CIFAR10Dataset(CIFARDataset):
    _CIFAR_TYPE = "cifar10"
    _DEFAULT_N_TASKS = 5


@register_dataset("cifar100")
class CIFAR100Dataset(CIFARDataset):
    _CIFAR_TYPE = "cifar100"
    _DEFAULT_N_TASKS = 20
