from typing import Callable, Tuple
import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class BaseDataset(Dataset):
    @property
    def task_id(self) -> int:
        return self._current_task

    @task_id.setter
    def task_id(self, value: int) -> None:
        self._current_task = value

    @property
    def n_classes_per_task(self) -> int:
        return self._n_classes_per_task

    @n_classes_per_task.setter
    def n_classes_per_task(self, value: int) -> None:
        self._n_classes_per_task = value

    @property
    def normalize_targets(self) -> bool:
        return self._normalize_targets

    @n_classes_per_task.setter
    def normalize_targets(self, value: bool) -> None:
        self._normalize_targets = value

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, int]:
        img, target = self.data[index], self.targets[index]

        if isinstance(img, Image.Image):
            pass
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif isinstance(img, str):
            img = Image.open(img).convert("RGB")
        else:
            raise ValueError("data type not supported")

        if self.transform is not None:
            img = self.transform(img)

        if self.normalize_targets:
            target -= self.task_id * self.n_classes_per_task

        return img, target, self.task_id
