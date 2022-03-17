import copy
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseMethod

from torchcl.utils.registry_utils import import_all_modules


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

def get_method(config: Dict[str, Any], *args, **kwargs):
    """Builds a method from a config.

    This assumes a 'name' key in the config which is used to determine what
    method class to instantiate. For instance, a config `{"name": "my_method",
    "foo": "bar"}` will find a class that was registered as "my_method"
    (see :func:`register_method`) and call .from_config on it.

    Args:
        config ([type]): [description]
    """

    assert config["method"] in METHOD_REGISTRY, "unknown method"
    method = METHOD_REGISTRY[config["method"]](config=config, *args, **kwargs)

    return method


# automatically import any Python files in the methods/ directory
import_all_modules(FILE_ROOT, "torchcl.methods")

from .finetuning import FineTuning
from .er import ER