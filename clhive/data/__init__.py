import traceback
from pathlib import Path

from ..utils.registry_utils import import_all_modules

from .continual_dataset import ContinualDataset


FILE_ROOT = Path(__file__).parent

DATASET_REGISTRY = {}
DATASET_REGISTRY_TB = {}
DATASET_CLASS_NAMES = set()
DATASET_CLASS_NAMES_TB = {}


def register_dataset(name, bypass_checks=False):
    """Registers a :class:`ContinualDataset` subclass.

    This decorator allows to instantiate a subclass of ContinualDataset 
    from a configuration file, even if the class itself is not
    part of the framework. To use it, apply this decorator to a
    ClassyDataset subclass like this:

    .. code-block:: python
      @register_dataset("my_dataset")
      class MyDataset(ContinualDataset):
          ...
    To instantiate a dataset from a configuration file, see
    :func:`build_dataset`.
    """

    def register_dataset_cls(cls):
        if not bypass_checks:
            if name in DATASET_REGISTRY:
                msg = "Cannot register duplicate dataset ({}). Already registered at \n{}\n"
                raise ValueError(msg.format(name, DATASET_REGISTRY_TB[name]))
            if not issubclass(cls, ContinualDataset):
                raise ValueError(
                    f"Dataset ({name}: {cls.__name__}) must extend ContinualDataset"
                )
            if cls.__name__ in DATASET_CLASS_NAMES:
                raise ValueError(
                    f"Cannot register dataset with duplicate class name({cls.__name__}). Previously registered at \n{DATASET_CLASS_NAMES_TB[cls.__name__]}\n"
                )
        tb = "".join(traceback.format_stack())
        DATASET_REGISTRY[name] = cls
        DATASET_CLASS_NAMES.add(cls.__name__)
        DATASET_REGISTRY_TB[name] = tb
        DATASET_CLASS_NAMES_TB[cls.__name__] = tb
        return cls

    return register_dataset_cls


def get_dataset(config, *args, **kwargs):
    """Builds a :class:`ContinualDataset` from a config.

    This assumes a 'name' key in the config which is used to determine what
    dataset class to instantiate. For instance, a config `{"name": "my_dataset",
    "folder": "/data"}` will find a class that was registered as "my_dataset"
    (see :func:`register_dataset`) and call .from_config on it.
    """
    dataset = DATASET_REGISTRY[config["dataset"]].from_config(config, *args, **kwargs)
    return dataset


# automatically import any Python files in the data/ directory
import_all_modules(FILE_ROOT, "clhive.data")

# import transforms
from .buffer import Buffer
from .cifar import CIFAR10, CIFAR100
from .mnist import MNISTDataset
