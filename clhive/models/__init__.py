import copy
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils.registry_utils import import_all_modules


FILE_ROOT = Path(__file__).parent

MODEL_REGISTRY = {}
MODEL_CLASS_NAMES = set()
MODEL_REGISTRY_TB = {}
MODEL_CLASS_NAMES_TB = {}


def register_model(name, bypass_checks=False):
    def register_model_cls(cls):
        if not bypass_checks:
            if name in MODEL_REGISTRY:
                raise ValueError(
                    f"Cannot register duplicate model ({name}). Already registered at \n{MODEL_REGISTRY_TB[name]}\n"
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


def auto_model(
    name: str,
    input_size: int,
    hidden_size: Optional[int] = None,
    output_size: Optional[int] = None,
    nf: Optional[int] = 32,
    **kwargs,
):

    assert name in MODEL_REGISTRY, "unknown model"
    return MODEL_REGISTRY[name](
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        nf=nf,
        **kwargs,
    )


# automatically import any Python files in the models/ directory
import_all_modules(FILE_ROOT, "clhive.models")

from .continual_model import ContinualModel
from .resnet import resnet18, resnet34, resnet50, resnet101
from .mlp import LinearClassifier, DistLinear, MLP
