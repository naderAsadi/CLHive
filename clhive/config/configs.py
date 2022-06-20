from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from typing import Dict, List, Optional, Tuple


@dataclass
class DataConfig:
    dataset: str = "cifar10"
    root: str = "../cl-datasets/data/"
    n_tasks: int = 5
    mem_size: int = 0
    download: Optional[bool] = True
    smooth: Optional[bool] = False
    n_workers: Optional[int] = 0
    transform: Optional[str] = "base"
    image_size: Optional[Tuple] = (3, 32, 32)
    n_classes: Optional[int] = 10
    n_classes_per_task: Optional[int] = 2


@dataclass
class MethodConfig:
    name: str = "finetuning"
    model: str = "resnet18"
    nf: int = 32
    use_snapshots: bool = False
    save_snapshots: bool = False
    snapshots_path: str = "./snapshots"


@dataclass
class TrainConfig:
    scenario: str = "single_head"
    batch_size: int = 32
    buffer_batch_size: int = 32
    n_epochs: int = 1
    n_iters: int = 1
    use_augs: bool = False


@dataclass
class EvalConfig:
    scenario: str = "single_head"
    batch_size: int = 32
    probe_eval: bool = False  # instead of keep_training_data
    cka_eval: bool = False
    eval_layer: Optional[int] = 4
    eval_every: Optional[int] = None
    n_epochs: int = 1


@dataclass
class OptimConfig:
    name: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0001
    lr_decay_rate: float = 0.1
    lr_decay_epochs: str = "120,160"


@dataclass
class Config:
    method: MethodConfig = MethodConfig()
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()
    optim: OptimConfig = OptimConfig()
