from typing import Optional
from dataclasses import dataclass
from torch import Tensor


@dataclass
class ModelOutput:
    logits: Optional[Tensor] = None
    hidden_states: Optional[Tensor] = None
    loss: Optional[Tensor] = None
