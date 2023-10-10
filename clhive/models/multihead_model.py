from typing import List
import torch
import torch.nn as nn

from . import get_model


class MultiHeadModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        n_tasks: int,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.n_classes_per_head = num_classes // n_tasks
        self.classifier = nn.ModuleList(
            [nn.Linear(backbone.hidden_dim, classes) for classes in classes_per_head]
        )

    @classmethod
    def auto_model(
        cls,
        name: str,
        num_classes: int,
        n_tasks: int,
        image_size: int,
        **kwargs,
    ) -> "MultiHeadModel":
        model = get_model(name=name, image_size=image_size, **kwargs)
        return cls(model=model, num_classes=num_classes, n_tasks=n_tasks)

    def forward_head(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pred = torch.zeros(
            x.size(0),
            self.n_classes_per_head,
            device=next(self.classifier.parameters()).device,
        )
        tasks = t.unique().tolist()
        for task in tasks:
            idx = t == task
            pred[idx] = self.classifier[task](x[idx, ...])

        return pred

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        outputs.logits = self.forward_head(outputs.hidden_states, t)

        return outputs
