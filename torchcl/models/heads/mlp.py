import torch
import torch.nn as nn
from torch.autograd import Variable

from torchcl.models.heads import register_head, BaseHead


def normalize(x):
    x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
    x_normalized = x.div(x_norm + 0.00001)
    return x_normalized

def add_linear(dim_in: int, dim_out: int, batch_norm: bool, relu: bool):
    layers = []
    layers.append(nn.Linear(dim_in, dim_out))
    if batch_norm:
        layers.append(nn.BatchNorm1d(dim_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


@register_head("linear")
class LinearClassifier(BaseHead):
    """Linear classifier"""
    def __init__(self, feature_dim, n_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feature_dim, n_classes)

    def forward(self, features):
        return self.fc(features)


@register_head("distlinear")
class DistLinear(BaseHead):
    def __init__(self, indim, outdim, weight=None):
        super(DistLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
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
        dim_in: int, 
        hidden_dim: int, 
        feature_dim: int, 
        num_layers: int = 2,
        batch_norm: bool = False
    ) -> None:
        """[summary]

        Args:
            dim_in (int): [description]
            hidden_dim (int): [description]
            feature_dim (int): [description]
            num_layers (int, optional): [description]. Defaults to 2.
            batch_norm (bool, optional): [description]. Defaults to False.
        """

        super(ProjectionMLP, self).__init__()
        self.layers = self._make_layers(dim_in, hidden_dim, feature_dim, num_layers, batch_norm)

    def _make_layers(self, dim_in, hidden_dim, feature_dim, num_layers, batch_norm):
        dims = [dim_in] + num_layers * [hidden_dim] + [feature_dim]
        layers = []
        for i in range(len(dims)):
            layers.append(add_linear(
                dim_in = dims[i], 
                dim_out = dims[i + 1], 
                batch_norm = batch_norm, 
                relu = (i < len(dims) - 2))
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
