import torch
import torch.nn as nn
from torch.autograd import Variable


def normalize(x):
    x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
    x_normalized = x.div(x_norm + 0.00001)
    return x_normalized

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet18', feat_dim=model_dict['resnet18'], num_classes=10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


class distLinear(nn.Module):
    def __init__(self, indim, outdim, weight=None):
        super(distLinear, self).__init__()
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


def add_linear(dim_in, dim_out, batch_norm, relu):
    layers = []
    layers.append(nn.Linear(dim_in, dim_out))
    if batch_norm:
        layers.append(nn.BatchNorm1d(dim_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class ProjectionMLP(nn.Module):
    def __init__(self, dim_in, hidden_dim, feat_dim, batch_norm, num_layers):
        super(ProjectionMLP, self).__init__()

        self.layers = self._make_layers(dim_in, hidden_dim, feat_dim, batch_norm, num_layers)

    def _make_layers(self, dim_in, hidden_dim, feat_dim, batch_norm, num_layers):
        layers = []
        layers.append(add_linear(dim_in, hidden_dim, batch_norm=batch_norm, relu=True))

        for _ in range(num_layers - 2):
            layers.append(add_linear(hidden_dim, hidden_dim, batch_norm=batch_norm, relu=True))
        
        layers.append(add_linear(hidden_dim, feat_dim, batch_norm=batch_norm, relu=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PredictionMLP(nn.Module):
    def __init__(self, dim_in, hidden_dim, out_dim):
        super(PredictionMLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)
