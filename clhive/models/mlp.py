import torch
import torch.nn as nn
from torch.autograd import Variable


def normalize(x):
    x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
    x_normalized = x.div(x_norm + 0.00001)
    return x_normalized


def add_linear(input_size: int, output_size: int, batch_norm: bool, relu: bool):
    layers = []
    layers.append(nn.Linear(input_size, output_size))
    if batch_norm:
        layers.append(nn.BatchNorm1d(output_size))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class LinearClassifier(nn.Module):
    """Linear classifier"""

    def __init__(self, input_size: int, output_size: int, **kwargs):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.output_size = output_size

    def forward(self, x):
        return self.fc(x)


class DistLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, weight=None, **kwargs):
        super().__init__()
        self.L = nn.Linear(input_size, output_size, bias=False)
        if weight is not None:
            self.L.weight.data = Variable(weight)

        self.scale_factor = 10
        self.output_size = output_size

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)

        L_norm = (
            torch.norm(self.L.weight, p=2, dim=1)
            .unsqueeze(1)
            .expand_as(self.L.weight.data)
        )
        L_normalized = self.L.weight.div(L_norm + 0.00001)

        cos_dist = torch.mm(x_normalized, L_normalized.transpose(0, 1))

        scores = self.scale_factor * (cos_dist)

        return scores


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        output_size: int,
        num_layers: int = 2,
        batch_norm: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.layers = self._make_layers(
            input_size, hidden_dim, output_size, num_layers, batch_norm
        )
        self.output_size = output_size

    def _make_layers(self, input_size, hidden_dim, output_size, num_layers, batch_norm):
        dims = [input_size] + num_layers * [hidden_dim] + [output_size]
        layers = []
        for i in range(len(dims)):
            layers.append(
                add_linear(
                    input_size=dims[i],
                    output_size=dims[i + 1],
                    batch_norm=batch_norm,
                    relu=(i < len(dims) - 2),
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
