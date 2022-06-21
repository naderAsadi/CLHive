import traceback
from pathlib import Path
from typing import Any, Optional, Union
import torch

from .base import BaseMethod
from ..models import ContinualModel
from ..utils.registry_utils import import_all_modules


FILE_ROOT = Path(__file__).parent

METHOD_REGISTRY = {}
METHOD_CLASS_NAMES = set()
METHOD_REGISTRY_TB = {}
METHOD_CLASS_NAMES_TB = {}


def register_method(name, bypass_checks=False):
    """Register a :class:`BaseMethod` subclass.

    This decorator allows instantiating a subclass of :class:`BaseMethod`
    from a configuration file. To use it, apply this decorator to a `BaseMethod`
    subclass.

    Args:
        name ([type]): [description]
        bypass_checks (bool, optional): [description]. Defaults to False.
    """

    def register_method_cls(cls):
        if not bypass_checks:
            if name in METHOD_REGISTRY:
                raise ValueError(
                    f"Cannot register duplicate method ({name}). Already registered at \n{METHOD_REGISTRY_TB[name]}\n"
                )
            if not issubclass(cls, BaseMethod):
                raise ValueError(
                    f"Method ({name}: {cls.__name__}) must extend BaseMethod"
                )
            if cls.__name__ in METHOD_CLASS_NAMES:
                raise ValueError(
                    f"Cannot register method with duplicate class name({cls.__name__}). Previously registered at \n{METHOD_CLASS_NAMES_TB[cls.__name__]}\n"
                )
        tb = "".join(traceback.format_stack())
        METHOD_REGISTRY[name] = cls
        METHOD_CLASS_NAMES.add(cls.__name__)
        METHOD_REGISTRY_TB[name] = tb
        METHOD_CLASS_NAMES_TB[cls.__name__] = tb
        return cls

    return register_method_cls


def auto_method(
    name: str,
    model: Union[ContinualModel, torch.nn.Module],
    optim: torch.optim,
    logger: Optional = None,
    **kwargs,
):
    assert name in METHOD_REGISTRY, "unknown method"
    return METHOD_REGISTRY[name](model=model, optim=optim, logger=logger)


# automatically import any Python files in the methods/ directory
import_all_modules(FILE_ROOT, "clhive.methods")

from .finetuning import FineTuning
from .er import ER
