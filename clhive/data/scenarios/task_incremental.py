from torch.utils.data import DataLoader

from .class_incremental import ClassIncrementalLoader
from ..sampler import ContinualSampler
from ..datasets import BaseDataset


class TaskIncrementalLoader(ClassIncrementalLoader):
    def __init__(
        self,
        dataset: BaseDataset,
        n_tasks: int,
        batch_size: int,
        n_workers: int = 0,
    ) -> None:
        super().__init__(
            dataset=dataset, n_tasks=n_tasks, batch_size=batch_size, n_workers=n_workers
        )

    def _create_dataloader(
        self, dataset: BaseDataset, batch_size: int, n_workers: int
    ) -> DataLoader:
        sampler = ContinualSampler(
            dataset=dataset,
            n_tasks=self.n_tasks,
            normalize_targets_per_task=True,
        )

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=n_workers,
            sampler=sampler,
        )
