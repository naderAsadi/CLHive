import copy
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from losses.base import BaseLoss

from utils.registry_utils import import_all_modules


FILE_ROOT = Path(__file__).parent

LOSS_REGISTRY = {}
LOSS_CLASS_NAMES = set()
LOSS_REGISTRY_TB = {}
LOSS_CLASS_NAMES_TB = {}

def register_loss(name, bypass_checks=False):
    """Register a :class:`BaseLoss` subclass.

    This decorator allows instantiating a subclass of :class:`BaseLoss`
    from a configuration file. To use it, apply this decorator to a `BaseLoss`
    subclass.

    Args:
        name ([type]): [description]
        bypass_checks (bool, optional): [description]. Defaults to False.
    """

    def register_loss_cls(cls):
        if not bypass_checks:
            if name in LOSS_REGISTRY:
                raise ValueError(
                    f"Cannot register duplicate loss ({name}). Already registered at \n{LOSS_REGISTRY_TB[name]}\n"
                )
            if not issubclass(cls, BaseLoss):
                raise ValueError(
                    f"Loss ({name}: {cls.__name__}) must extend BaseLoss"
                )
            if cls.__name__ in LOSS_CLASS_NAMES:
                raise ValueError(
                    f"Cannot register loss with duplicate class name({cls.__name__}). Previously registered at \n{LOSS_CLASS_NAMES_TB[cls.__name__]}\n"
                )
        tb = "".join(traceback.format_stack())
        LOSS_REGISTRY[name] = cls
        LOSS_CLASS_NAMES.add(cls.__name__)
        LOSS_REGISTRY_TB[name] = tb
        LOSS_CLASS_NAMES_TB[cls.__name__] = tb
        return cls
    
    return register_loss_cls

def get_loss(config: Dict[str, Any]):
    """Builds a loss from a config.

    This assumes a 'name' key in the config which is used to determine what
    loss class to instantiate. For instance, a config `{"name": "my_loss",
    "foo": "bar"}` will find a class that was registered as "my_loss"
    (see :func:`register_loss`) and call .from_config on it.

    Args:
        config ([type]): [description]
    """

    assert config["name"] in LOSS_REGISTRY, "unknown loss"
    loss = LOSS_REGISTRY[config["name"]].from_config(config)

    return loss


# automatically import any Python files in the losses/ directory
import_all_modules(FILE_ROOT, "losses")