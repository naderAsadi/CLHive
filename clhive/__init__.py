"""CLHive is a PyTorch framework for Continual Learning research.
"""

__name__ = "clhive"
__version__ = "0.0.1"

from clhive import config, data, loggers, methods, models, scenarios, utils

from clhive.config import config_parser
from clhive.data import ReplayBuffer
from clhive.models import auto_model
from clhive.methods import auto_method
from clhive.utils import evaluators
from clhive.utils.trainer import Trainer
