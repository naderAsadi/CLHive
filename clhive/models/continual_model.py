from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import Config


class ContinualModel(nn.Module):
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
        self, backbone: nn.Module, heads: nn.ModuleDict, scenario: str = "single_head",
    ) -> None:
        """[summary]

        Args:
            backbone (nn.Module): [description]
            heads (nn.ModuleDict): [description]
            scenario (str): Supported CL scenarios are `single_head` or `multi_head`
        """

        assert scenario in [
            "single_head",
            "multi_head",
        ], f"Supported CL scenarios are `single_head` or `multi_head`, but {scenario} is entered."

        super(ContinualModel, self).__init__()
        self.backbone = backbone
        self.heads = heads
        self.scenario = scenario

        if self.scenario is "single_head":
            self._pred_dim = self.heads.out_dim
        else:
            self._pred_dim = list(self.heads.values())[0].out_dim

    @classmethod
    def from_config(cls, config: Config) -> "ContinualModel":
        """Instantiates a Model from a configuration.

        Args:
            config (Dict): A configuration for the Model.

        Returns:
            A torch.nn.Module instance.
        """

        return cls(*args, **kwargs)

    def set_heads(self, heads: nn.ModuleDict):
        self.heads = heads

    def add_head(self, name: str, head: nn.Module):
        self.heads.update({name: head})

    def forward_model(self, x):
        return self.backbone(x)

    def forward_head(self, x, task: Optional[int] = None):
        if self.scenario is "single_head":
            return self.heads[str(task)](x)

        pred = torch.zeros(x.size(0), self._pred_dim).to(x.get_device())
        tasks = task.unique().tolist()
        for t in tasks:
            idx = task == t
            pred[idx] = self.heads[str(t)](x[idx])
        return pred

    def forward(self, x, task: Optional[int] = None):
        """
        Perform computation of blocks in the order define in get_blocks.
        """
        if self.scenario is not "single_head":
            assert (
                task not in self.heads.keys()
            ), f"{task} does not exist in {self.heads.keys()}"

        x = self.forward_model(x)
        x = self.forward_head(x, task)
        return x
