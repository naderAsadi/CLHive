import sys

import hydra
from omegaconf import DictConfig


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
    with hydra.initialize(config_path=config_path, job_name=job_name):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    return cfg
