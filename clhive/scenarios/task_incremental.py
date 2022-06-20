from types import Optional

from .class_incremental import ClassIncremental


class TaskIncremental(ClassIncremental):

    def __init__(
        self,
        dataset,
        n_tasks: int,
        batch_size: int,
        n_workers: Optional[int] = 0,
        smooth_task_boundary: Optional[bool] = False,
    ) -> "TaskIncremental":
        
        super().__init__(dataset, n_tasks, batch_size, n_workers, smooth_task_boundary)

    def _get_dataloader(self, dataset) -> DataLoader:
        sampler = ContinualSampler(
            dataset=dataset,
            n_tasks=self.n_tasks,
            smooth_boundary=self.smooth_task_boundary,
            normal_targets_per_task=True,
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            sampler=sampler,
        )