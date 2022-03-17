from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class ModelWrapper(nn.Module):
    """Base class for models in torchCL.

    A model refers either to a specific architecture (e.g. ResNet50) or a
    family of architectures (e.g. ResNet). Models can take arguments in the
    constructor in order to configure different behavior (e.g.
    hyperparameters).  torchCL models must implement :func:`from_config` in
    order to allow instantiation from a configuration file. Like regular
    PyTorch models, models must also implement :func:`forward`, where
    the bulk of the inference logic lives.

    Models also have some advanced functionality for production
    fine-tuning systems. For example, we allow users to train a trunk
    model and then attach heads to the model via the attachable
    blocks.  Making your model support the trunk-heads paradigm is
    completely optional.
    """

    def __init__(
        self, 
        model: nn.Module, 
        heads: nn.ModuleDict
    ) -> None:
        """[summary]

        Args:
            model (nn.Module): [description]
            heads (nn.ModuleDict): [description]
        """

        super(ModelWrapper, self).__init__()
        self._model = model
        self._heads = heads
        self._pred_dim = list(self._heads.values())[0].out_dim
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], *args, **kwargs) -> "ModelWrapper":
        """Instantiates a Model from a configuration.

        Args:
            config (Dict): A configuration for the Model.

        Returns:
            A torch.nn.Module instance.
        """

        return cls(*args, **kwargs)
    
    def set_heads(self, heads: nn.ModuleDict):
        self._heads = heads
    
    def add_head(self, name: str, head: nn.Module):
        self._heads.update({name: head})
    
    def forward_model(self, x):
        return self._model(x)

    def forward_head(self, x, task: int):
        if isinstance(task, int):
            return self._heads[str(task)](x)
        
        pred = torch.zeros(x.size(0), self._pred_dim).to(x.get_device())
        tasks = task.unique().tolist()
        for t in tasks:
            idx = (task == t)
            pred[idx] = self._heads[str(t)](x[idx])
        return pred

    def forward(self, x, task: int):
        """
        Perform computation of blocks in the order define in get_blocks.
        """
        assert task not in self._heads.keys(), (
            f"{task} does not exist in {self._heads.keys()}"
        )

        x = self.forward_model(x)
        x = self.forward_head(x, task)
        return x
        

        

        
