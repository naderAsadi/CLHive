from typing import Optional
from dataclasses import dataclass


@dataclass
class DataConfig:
    dataset: str
    root: str
    num_classes: int
    image_size: int
    buffer_capacity: int = 0
    num_workers: int = 0
