from typing import Optional

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
        """_summary_

        Args:
            dataset (_type_): _description_
            n_tasks (int): _description_
            batch_size (int): _description_
            n_workers (Optional[int], optional): _description_. Defaults to 0.
            smooth_task_boundary (Optional[bool], optional): _description_. Defaults to False.

        Returns:
            TaskIncremental: _description_
        """
        super().__init__(dataset, n_tasks, batch_size, n_workers, smooth_task_boundary)

        # Normalize labels per task for task-incremental scenario
        self.dataset.normalize_targets_per_task = True
