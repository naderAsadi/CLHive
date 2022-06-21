from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from .continual_sampler import ContinualSampler


class ClassIncremental:
    def __init__(
        self,
        dataset,
        n_tasks: int,
        batch_size: int,
        n_workers: Optional[int] = 0,
        smooth_task_boundary: Optional[bool] = False,
    ) -> "ClassIncremental":
        """_summary_

        Args:
            dataset (_type_): _description_
            n_tasks (int): _description_
            batch_size (int): _description_
            n_workers (Optional[int], optional): _description_. Defaults to 0.
            smooth_task_boundary (Optional[bool], optional): _description_. Defaults to False.

        Returns:
            ClassIncremental: _description_
        """

        self.dataset = dataset
        self.n_tasks = n_tasks
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.smooth_task_boundary = smooth_task_boundary

        self._task_id = 0
        self.loader = self._create_dataloader()

        self.dataset.normalize_targets_per_task = False

    @property
    def n_samples(self) -> int:
        """Total number of samples in the whole continual setting."""
        return len(self.dataset)

    @property
    def n_classes(self) -> int:
        """Total number of classes in the whole continual setting."""
        if isinstance(self.dataset, torch.utils.data.Subset):
            targets = np.array(self.dataset.dataset.targets)[self.dataset.indices]
        else:
            targets = self.dataset.dataset.targets

        return len(np.unique(targets))

    def __len__(self) -> int:
        """Returns the number of tasks.

        Returns:
            int: Number of tasks.
        """
        return self.n_tasks

    def __iter__(self):
        """Used for iterating through all tasks with the CLLoader in a for loop."""
        self._task_id = 0
        return self

    def __next__(self):
        if self._task_id >= len(self):
            raise StopIteration
        self.loader.sampler.set_task(self._task_id)
        self._task_id += 1

        return self.loader

    def __getitem__(self, task_id: int):
        self.loader.sampler.set_task(task_id)
        return self.loader

    def _create_dataloader(self) -> DataLoader:
        sampler = ContinualSampler(
            dataset=self.dataset,
            n_tasks=self.n_tasks,
            smooth_boundary=self.smooth_task_boundary,
            normal_targets_per_task=False,
        )
        self.dataset.n_classes_per_task = sampler.cpt

        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            sampler=sampler,
        )
