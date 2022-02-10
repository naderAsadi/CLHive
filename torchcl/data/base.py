from typing import Any, Callable, Dict, Optional, Sequence, Union

from torchcl.data.transforms.base_transform import BaseTransform


class BaseDataset:

    def __init__(
        self,
        dataset: Sequence,
        transform: Optional[Union[BaseTransform, Callable]]
    ) -> None:
        """Constructor for BaseDataset

        Args:
            dataset (Sequence): [description]
            transform (Optional[Union[BaseTransform, Callable]]): [description]
        """

        self.dataset = dataset
        self.transform = transform
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseDataset":
        """Instantiates a Dataset from a configuration.

        Args:
            config: A configuration for the Dataset.
        """

        raise NotImplementedError

    def __getitem__(self, index: int):
        assert index >= 0 and index < len(self.dataset), f"Provided index ({index}) is outside of dataset range."
        sample = self.dataset[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return len(self.dataset)