from typing import Any, Callable, Dict, List

import torch


def get_optimizer(config: Dict[str, Any]):
    assert hasattr(torch.optim, config.name), (
        f"{config.name} is not a registered optimizer in torch.optim"
    )
    optim = getattr(torch.optim, config.name)(**config)
    return optim

