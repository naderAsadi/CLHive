import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class DataSequenceLoader:
    def __init__(self, datasets: List[Dataset], batch_size: int, n_workers: int):
        self.datasets = datasets

    def __len__(self) -> int:
        return len(self.datasets)

    def __next__(self):
        pass

    def __getitem__(self, task_id: int):
        pass
