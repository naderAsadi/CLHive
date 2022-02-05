import copy
import os
import time
from typing import Any, List, Optional, Tuple, Union

import torch

from torchcl.models import ModelWrapper


class Trainer:
    def __init__(
        self,
        model: ModelWrapper
    ) -> None:
        
        self.model = model