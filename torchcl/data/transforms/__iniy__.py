import copy
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List

import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

from utils.registry_utils import import_all_modules


FILE_ROOT = Path(__file__).parent

TRANSFORM_REGISTRY = {}
TRANSFORM_REGISTRY_TB = {}

def register_transform(name: str, bypass_checks=False):
    """Registers a :class:`BaseTransform` subclass.

    This decorator allows Classy Vision to instantiate a subclass of
    :class:`BaseTransform` from a configuration file, even if the class itself is not
    part of the framework. To use it, apply this decorator to a
    BaseTransform subclass like this:
    .. code-block:: python
      @register_transform("my_transform")
      class MyTransform(BaseTransform):
          ...
    """

    def register_transform_cls(cls: Callable[..., Callable]):
        if not bypass_checks:
            if name in TRANSFORM_REGISTRY:
                msg = "Cannot register duplicate transform ({}). Already registered at \n{}\n"
                raise ValueError(msg.format(name, TRANSFORM_REGISTRY_TB[name]))
            if hasattr(transforms, name) or hasattr(transforms_video, name):
                raise ValueError(
                    "{} has existed in torchvision.transforms, Please change the name!".format(
                        name
                    )
                )
        TRANSFORM_REGISTRY[name] = cls
        tb = "".join(traceback.format_stack())
        TRANSFORM_REGISTRY_TB[name] = tb
        return cls

    return register_transform_cls


def get_transform(transform_config: Dict[str, Any]) -> Callable:
    """Builds a :class:`BaseTransform` from a config.

    This assumes a 'name' key in the config which is used to determine what
    transform class to instantiate. For instance, a config `{"name":
    "my_transform", "foo": "bar"}` will find a class that was registered as
    "my_transform" and call .from_config on it.

    In addition to transforms registered with :func:`register_transform`, we
    also support instantiating transforms available in the
    `torchvision.transforms <https://pytorch.org/docs/stable/torchvision/
    transforms.html>`_ module. Any keys in the config will get expanded
    to parameters of the transform constructor. For instance, the following
    call will instantiate a :class:`torchvision.transforms.CenterCrop`:
    
    .. code-block:: python
      get_transform({"name": "CenterCrop", "size": 224})
    """
    assert (
        "name" in transform_config
    ), f"name not provided for transform: {transform_config}"
    name = transform_config["name"]

    transform_args = {k: v for k, v in transform_config.items() if k != "name"}

    if name in TRANSFORM_REGISTRY:
        transform = TRANSFORM_REGISTRY[name].from_config(transform_args)
    else:
        # the name should be available in torchvision.transforms
        # if users specify the torchvision transform name in snake case,
        # we need to convert it to title case.
        if not (hasattr(transforms, name) or hasattr(transforms_video, name)):
            name = name.title().replace("_", "")
        assert hasattr(transforms, name) or hasattr(transforms_video, name), (
            f"{name} isn't a registered tranform"
            ", nor is it available in torchvision.transforms"
        )
        if hasattr(transforms, name):
            transform = getattr(transforms, name)(**transform_args)
        else:
            transform = getattr(transforms_video, name)(**transform_args)

    return transform

def get_transforms(transforms_config: List[Dict[str, Any]]) -> Callable:
    """
    Builds a transform from the list of transform configurations.
    """
    transform_list = [get_transform(config) for config in transforms_config]
    return transforms.Compose(transform_list)


# automatically import any Python files in the transforms/ directory
import_all_modules(FILE_ROOT, "classy_vision.dataset.transforms")

from data.transforms.base_transform import BaseTransform
from data.transforms.simclr_transform import SimCLRTransform
