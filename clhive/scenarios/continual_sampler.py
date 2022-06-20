import numpy as np
import torch
from torch.utils.data import Sampler


class ContinualSampler(Sampler):
    def __init__(
        self,
        dataset,
        n_tasks: int,
        smooth_boundary: Optional[bool] = False,
        normal_targets_per_task: Optional[bool] = False,
    ):
        """[summary]

        Args:
            dataset: [description]
            n_tasks (int): [description]
            smooth_boundary (bool, optional): [description]. Defaults to False.
        """

        self.ds = dataset
        self.n_tasks = n_tasks
        self.smooth_boundary = smooth_boundary
        self.normal_targets_per_task = normal_targets_per_task

        # print(dataset.data, dataset.targets)
        if isinstance(dataset, torch.utils.data.Subset):
            ds_targets = np.array(dataset.dataset.targets)[dataset.indices]
        else:
            ds_targets = dataset.dataset.targets

        self.classes = np.unique(ds_targets)

        ds_targets = np.array(ds_targets)
        self.n_samples = ds_targets.shape[0]
        self.n_classes = self.classes.shape[0]

        assert (
            self.n_classes % n_tasks == 0
        ), f"Cannot break {self.n_classes} classes into {n_tasks} tasks."
        self.cpt = self.n_classes // n_tasks

        self.sample_all_seen_tasks = False

        self._current_task = None
        self.target_indices = {}

        # for smooth datasets
        self.t = 0
        self.per_class_samples_left = torch.zeros(self.classes.shape[0]).int()

        for label in self.classes:
            self.target_indices[label] = np.squeeze(np.argwhere(ds_targets == label))
            np.random.shuffle(self.target_indices[label])
            self.per_class_samples_left[label] = self.target_indices[label].shape[0]

    @property
    def current_task(self):
        return self._current_task

    def set_task(self, task_id: int, sample_all_seen_tasks: bool = False):
        self._current_task = task_id
        self.ds._set_task(task_id)
        self.sample_all_seen_tasks = sample_all_seen_tasks

    def _fetch_task_samples(self, task_id):
        if self.sample_all_seen_tasks:
            task_classes = self.classes[: self.cpt * (task_id + 1)]
        else:
            task_classes = self.classes[self.cpt * task_id : self.cpt * (task_id + 1)]

        task_samples = []

        for t in task_classes:
            t_indices = self.target_indices[t]
            task_samples += [t_indices]

        task_samples = np.concatenate(task_samples)
        np.random.shuffle(task_samples)

        self.task_samples = task_samples

    def __iter__(self):
        self._fetch_task_samples(self._current_task)
        for item in self.task_samples:
            yield item

    def __len__(self):
        samples_per_task = self.n_samples // self.n_tasks
        return samples_per_task