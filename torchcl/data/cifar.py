from typing import Any, Callable, Dict, Optional, Sequence, Union

from torchvision.datasets.cifar import CIFAR10, CIFAR100

from data import register_dataset
from data.base import BaseDataset
from data.transforms import get_transform


class CIFARDataset(BaseDataset):

    _CIFAR_TYPE = None
    _MEAN = (0.4914, 0.4822, 0.4465)
    _STD  = (0.2023, 0.1994, 0.2010)

    def __init__(
        self,
        root: str, 
        transform: Optional[Union[BaseTransform, Callable]],
        train: bool,
        download: bool = True
    ) -> None:

        assert self._CIFAR_TYPE in [
            "cifar10",
            "cifar100"
        ], "CIFARDataset must be subclassed and a valid _CIFAR_TYPE provided"
        if self._CIFAR_TYPE == "cifar10":
            dataset = CIFAR10(root, train=train, download=download)
        if self._CIFAR_TYPE == "cifar100":
            dataset = CIFAR100(root, train=train, download=download)

        super().__init__(
            dataset=dataset, transform=transform
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CIFARDataset":
        """Instantiates a CIFARDataset from a configuration.
        Args:
            config: A configuration for a CIFARDataset.
                See :func:`__init__` for parameters expected in the config.
        Returns:
            A CIFARDataset instance.
        """
        
        root = config.get("root")
        train = config.get("train")
        transform_config = config.get("transforms")
        download = config.get("download")

        transform = get_transform(transform_config)
        return cls(
            root=root, 
            transform=transform,
            train=train,
            download=True
        )


@register_dataset("classy_cifar10")
class CIFAR10Dataset(CIFARDataset):
    _CIFAR_TYPE = "cifar10"


@register_dataset("classy_cifar100")
class CIFAR100Dataset(CIFARDataset):
    _CIFAR_TYPE = "cifar100"
