import copy
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List

from torch.utils.data import Sampler
# import torchvision.transforms as transforms
# import torchvision.transforms._transforms_video as transforms_video

from torchcl.utils.registry_utils import import_all_modules


FILE_ROOT = Path(__file__).parent

SAMPLER_REGISTRY = {}
SAMPLER_REGISTRY_TB = {}

def register_sampler(name: str, bypass_checks=False):
    """Registers a :class:`torch.utils.data.Sampler` subclass.

    This decorator allows to instantiate a subclass of
    :class:`torch.utils.data.Sampler` from a configuration file, 
    even if the class itself is not part of the framework. 
    """

    def register_sampler_cls(cls: Callable[..., Callable]):
        if not bypass_checks:
            if name in SAMPLER_REGISTRY:
                msg = "Cannot register duplicate sampler ({}). Already registered at \n{}\n"
                raise ValueError(msg.format(name, SAMPLER_REGISTRY_TB[name]))
        SAMPLER_REGISTRY[name] = cls
        tb = "".join(traceback.format_stack())
        SAMPLER_REGISTRY_TB[name] = tb
        return cls

    return register_sampler_cls


def get_sampler(sampler_config: Dict[str, Any]) -> Callable:
    
    assert sampler_config["name"] in SAMPLER_REGISTRY, "unknown sampler"
    sampler = SAMPLER_REGISTRY[name](sampler_config)

    return sampler


# automatically import any Python files in the samplers/ directory
import_all_modules(FILE_ROOT, "torchcl.data.samplers")

from torchcl.data.samplers.continual_sampler import ContinualSampler