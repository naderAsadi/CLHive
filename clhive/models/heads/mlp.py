import torch
import torch.nn as nn
from torch.autograd import Variable

from . import register_head, BaseHead


def normalize(x):
    x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
    x_normalized = x.div(x_norm + 0.00001)
    return x_normalized

def add_linear(in_dim: int, out_dim: int, batch_norm: bool, relu: bool):
    layers = []
    layers.append(nn.Linear(in_dim, out_dim))
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_dim))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


@register_head("linear")
class LinearClassifier(BaseHead):
    """Linear classifier"""
    def __init__(self, feature_dim, n_classes):
        super(LinearClassifier, self).__init__(feature_dim, n_classes)
        self.fc = nn.Linear(feature_dim, n_classes)

    def forward(self, features):
        return self.fc(features)


@register_head("distlinear")
class DistLinear(BaseHead):
    def __init__(self, in_dim, out_dim, weight=None):
        super(DistLinear, self).__init__(in_dim, out_dim)
        self.L = nn.Linear( in_dim, out_dim, bias = False)
        if weight is not None:
            self.L.weight.data = Variable(weight)

        self.scale_factor = 10

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)

        L_norm = torch.norm(self.L.weight, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
        L_normalized = self.L.weight.div(L_norm + 0.00001)

        cos_dist = torch.mm(x_normalized, L_normalized.transpose(0,1))

        scores = self.scale_factor * (cos_dist)

        return scores


@register_head("projection")
class ProjectionMLP(BaseHead):
    def __init__(
        self, 
        in_dim: int, 
        hidden_dim: int, 
        out_dim: int, 
        num_layers: int = 2,
        batch_norm: bool = False
    ) -> None:
        """[summary]

        Args:
            in_dim (int): [description]
            hidden_dim (int): [description]
            out_dim (int): [description]
            num_layers (int, optional): [description]. Defaults to 2.
            batch_norm (bool, optional): [description]. Defaults to False.
        """

        super(ProjectionMLP, self).__init__(in_dim, out_dim)
        self.layers = self._make_layers(in_dim, hidden_dim, out_dim, num_layers, batch_norm)

    def _make_layers(self, in_dim, hidden_dim, out_dim, num_layers, batch_norm):
        dims = [in_dim] + num_layers * [hidden_dim] + [out_dim]
        layers = []
        for i in range(len(dims)):
            layers.append(add_linear(
                in_dim = dims[i], 
                out_dim = dims[i + 1], 
                batch_norm = batch_norm, 
                relu = (i < len(dims) - 2))
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
