import os
import sys
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from .configs import Config, MethodConfig, DataConfig, TrainConfig, EvalConfig, OptimConfig


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

def config_parser(
    config_path:str, 
    config_name:str, 
    job_name:str
) -> DictConfig:
    """[summary]

    Args:
        config_path (str): [description]
        config_name (str): [description]
        job_name (str): [description]

    Returns:
        DictConfig: [description]
    """

    overrides = sys.argv[1:]
    hydra.initialize_config_dir(config_dir=os.path.abspath(config_path), job_name=job_name)
    cfg = hydra.compose(config_name=config_name, overrides=overrides)
    return cfg
