import numpy as np
import torch
from torch.utils.data import DataLoader

from ..sampler import ContinualSampler
from ..datasets import BaseDataset


class ClassIncrementalLoader:
    def __init__(
        self,
        dataset: BaseDataset,
        n_tasks: int,
        batch_size: int,
        n_workers: int = 0,
    ) -> None:
        self.dataset = dataset
        self.n_tasks = n_tasks
        self.loader = self._create_dataloader(
            dataset=dataset, batch_size=batch_size, n_workers=n_workers
        )

    @property
    def n_samples(self) -> int:
        """Total number of samples in the whole continual setting."""
        return len(self.dataset)

    @property
    def n_classes(self) -> int:
        """Total number of classes in the whole continual setting."""
        return len(np.unique(self.dataset.targets))

    def __len__(self) -> int:
        """Returns the number of tasks."""
        return self.n_tasks

    def __iter__(self):
        """Used for iterating through all tasks with the CLLoader in a loop."""
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

    def _create_dataloader(
        self, dataset: BaseDataset, batch_size: int, n_workers: int
    ) -> DataLoader:
        sampler = ContinualSampler(
            dataset=dataset,
            n_tasks=self.n_tasks,
            normalize_targets_per_task=False,
        )

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=n_workers,
            sampler=sampler,
        )
