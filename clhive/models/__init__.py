import copy
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .model_wrapper import ModelWrapper
from ..utils.registry_utils import import_all_modules


FILE_ROOT = Path(__file__).parent

MODEL_REGISTRY = {}
MODEL_CLASS_NAMES = set()
MODEL_REGISTRY_TB = {}
MODEL_CLASS_NAMES_TB = {}

def register_model(name, bypass_checks=False):
    """Register a :class:`ModelWrapper` subclass.

    This decorator allows instantiating a subclass of :class:`ModelWrapper`
    from a configuration file. To use it, apply this decorator to a `ModelWrapper`
    subclass.

    Args:
        name ([type]): [description]
        bypass_checks (bool, optional): [description]. Defaults to False.
    """

    def register_model_cls(cls):
        if not bypass_checks:
            if name in MODEL_REGISTRY:
                raise ValueError(
                    f"Cannot register duplicate model ({name}). Already registered at \n{MODEL_REGISTRY_TB[name]}\n"
                )
            if not issubclass(cls, ModelWrapper):
                raise ValueError(
                    f"Model ({name}: {cls.__name__}) must extend ModelWrapper"
                )
            if cls.__name__ in MODEL_CLASS_NAMES:
                raise ValueError(
                    f"Cannot register model with duplicate class name({cls.__name__}). Previously registered at \n{MODEL_CLASS_NAMES_TB[cls.__name__]}\n"
                )
        tb = "".join(traceback.format_stack())
        MODEL_REGISTRY[name] = cls
        MODEL_CLASS_NAMES.add(cls.__name__)
        MODEL_REGISTRY_TB[name] = tb
        MODEL_CLASS_NAMES_TB[cls.__name__] = tb
        return cls
    
    return register_model_cls

def get_model(config: Dict[str, Any], *args, **kwargs):
    """Builds a model from a config.

    This assumes a 'name' key in the config which is used to determine what
    model class to instantiate. For instance, a config `{"name": "my_model",
    "foo": "bar"}` will find a class that was registered as "my_model"
    (see :func:`register_model`) and call .from_config on it.

    Args:
        config ([type]): [description]
    """

    assert config["name"] in MODEL_REGISTRY, "unknown model"
    model = MODEL_REGISTRY[config["name"]].from_config(config, *args, **kwargs)

    return model


# automatically import any Python files in the models/ directory
import_all_modules(FILE_ROOT, "clhive.models")

from .model_wrapper import ModelWrapper
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101
from .heads import *