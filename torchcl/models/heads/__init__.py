import copy
import traceback
from pathlib import Path

from utils.registry_utils import import_all_modules

from model.heads.base import BaseHead


FILE_ROOT = Path(__file__).parent


HEAD_REGISTRY = {}
HEAD_CLASS_NAMES = set()
HEAD_REGISTRY_TB = {}
HEAD_CLASS_NAMES_TB = {}


def register_head(name, bypass_checks=False):
    """Registers a BaseHead subclass.

    This decorator allows to instantiate a subclass of BaseHead 
    from a configuration file, even if the class itself is not
    part of the framework. To use it, apply this decorator to a
    BaseHead subclass, like this:

    .. code-block:: python
      @register_head("my_head")
      class MyHead(BaseHead):
          ...
    """

    def register_head_cls(cls):
        if not bypass_checks:
            if name in HEAD_REGISTRY:
                raise ValueError(
                    f"Cannot register duplicate head ({name}). Already registered at \n{HEAD_REGISTRY_TB[name]}\n"
                )
            if not issubclass(cls, BaseHead):
                raise ValueError(
                    f"Head ({name}: {cls.__name__}) must extend BaseHead"
                )
            if cls.__name__ in HEAD_CLASS_NAMES:
                raise ValueError(
                    f"Cannot register head with duplicate class name({cls.__name__}). Previously registered at \n{HEAD_CLASS_NAMES_TB[cls.__name__]}\n"
                )
        tb = "".join(traceback.format_stack())
        HEAD_REGISTRY[name] = cls
        HEAD_CLASS_NAMES.add(cls.__name__)
        HEAD_REGISTRY_TB[name] = tb
        HEAD_CLASS_NAMES_TB[cls.__name__] = tb
        return cls

    return register_head_cls


def get_head(config):
    """Builds a Head from a config.

    This assumes a 'name' key in the config which is used to determine what
    head class to instantiate. For instance, a config `{"name": "my_head",
    "foo": "bar"}` will find a class that was registered as "my_head"
    (see :func:`register_head`) and call .from_config on it."""

    return HEAD_REGISTRY[name].from_config(config)


# automatically import any Python files in the heads/ directory
import_all_modules(FILE_ROOT, "models.heads")