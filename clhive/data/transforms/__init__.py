import copy
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List

import torchvision.transforms as pth_transforms

from ...utils.registry_utils import import_all_modules


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
            if hasattr(pth_transforms, name):
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


def get_transform(transform_name: str) -> Callable:
    """Builds a :class:`BaseTransform` from transform name.

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

    if transform_name in TRANSFORM_REGISTRY:
        transform = TRANSFORM_REGISTRY[transform_name]()
    else:
        # the name should be available in torchvision.transforms
        # if users specify the torchvision transform name in snake case,
        # we need to convert it to title case.
        if not (hasattr(pth_transforms, transform_name)):
            transform_name = transform_name.title().replace("_", "")
        assert hasattr(pth_transforms, transform_name), (
            f"{name} isn't a registered tranform"
            ", nor is it available in torchvision.transforms"
        )
        transform = getattr(pth_transforms, name)()

    return transform


# automatically import any Python files in the transforms/ directory
import_all_modules(FILE_ROOT, "clhive.data.transforms")

from .simclr_transform import SimCLRTransform
from .base_transform import BaseTransform
