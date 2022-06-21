from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
import torch
from torchvision import transforms


class ContinualDataset:
    def __init__(
        self,
        dataset: Sequence,
        transform: Optional[Callable] = None,
        normalize_targets_per_task: Optional[bool] = False,
    ) -> None:
        """Constructor for ContinualDataset

        Args:
            dataset (Sequence): [description]
            transform (Callable): [description]
            normalize_targets_per_task (bool): [description]
        """
        self.dataset = dataset
        self.transform = transform
        self.normalize_targets_per_task = normalize_targets_per_task

        self.n_classes_per_task = None
        self._current_task = 0

        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        assert index >= 0 and index < len(
            self.dataset
        ), f"Provided index ({index}) is outside of dataset range."

        sample = self.dataset[index]
        data, targets = sample

        data = self.transform(data)

        if self.normalize_targets_per_task and self.n_classes_per_task:
            targets -= self._current_task * self.n_classes_per_task

        return data, targets, self._current_task

    def _set_task(self, task_id: int):
        self._current_task = task_id
