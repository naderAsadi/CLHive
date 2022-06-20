from typing import Any, Callable, Dict, Optional, Sequence, Union

from ..transforms.base_transform import BaseTransform


class BaseDataset:
    def __init__(
        self, dataset: Sequence, transform: Optional[Union[BaseTransform, Callable]]
    ) -> None:
        """Constructor for BaseDataset

        Args:
            dataset (Sequence): [description]
            transform (Optional[Union[BaseTransform, Callable]]): [description]
        """
        if transform is None:
            transform = BaseTransform()

        self.dataset = dataset
        self.transform = transform
        self._current_task = 0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseDataset":
        """Instantiates a Dataset from a configuration.

        Args:
            config: A configuration for the Dataset.
        """

        raise NotImplementedError

    def _set_task(self, task_id: int):
        self._current_task = task_id

    def __getitem__(self, index: int):
        assert index >= 0 and index < len(
            self.dataset
        ), f"Provided index ({index}) is outside of dataset range."
        sample = self.dataset[index]
        data, targets = sample
        if self.transform is not None:
            data = self.transform(data)
        return data, targets, self._current_task

    def __len__(self):
        return len(self.dataset)
