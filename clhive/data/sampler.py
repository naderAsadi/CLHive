import numpy as np
import torch
from torch.utils.data import Sampler

from .datasets import BaseDataset


class ContinualSampler(Sampler):
    def __init__(
        self,
        dataset: BaseDataset,
        n_tasks: int,
        normalize_targets_per_task: bool = False,
    ):
        self.dataset = dataset
        self.n_tasks = n_tasks

        assert hasattr(dataset, "data") and hasattr(
            dataset, "targets"
        ), "dataset object has no attribute `data` and `targets`"
        ds_targets = dataset.targets

        self.classes = np.unique(ds_targets)
        ds_targets = np.array(ds_targets)
        self.n_samples = ds_targets.shape[0]
        n_classes = self.classes.shape[0]

        assert (
            n_classes % n_tasks == 0
        ), f"Cannot break {n_classes} classes into {n_tasks} tasks."
        self.cpt = n_classes // n_tasks

        self.target_indices = {}
        self.per_class_samples_left = torch.zeros(self.classes.shape[0]).int()
        for label in self.classes:
            self.target_indices[label] = np.squeeze(np.argwhere(ds_targets == label))
            np.random.shuffle(self.target_indices[label])
            self.per_class_samples_left[label] = self.target_indices[label].shape[0]

        # Defaults
        self._sample_all_seen_tasks = False
        self._current_task = 0
        self.dataset.task_id = self._current_task
        self.dataset.n_classes_per_task = self.cpt
        self.dataset.normalize_targets = normalize_targets_per_task

    @property
    def current_task(self):
        return self._current_task

    def set_task(self, task_id: int, sample_all_seen_tasks: bool = False):
        self._current_task = task_id
        self.dataset.task_id = task_id
        self._sample_all_seen_tasks = sample_all_seen_tasks

    def _fetch_task_samples(self, task_id):
        if self._sample_all_seen_tasks:
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
