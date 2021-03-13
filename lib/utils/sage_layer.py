import math

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


class SAGEGraphLayer(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(SAGEGraphLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(2*in_features, out_features)

        self.relu = nn.ReLU()

    def forward(self, input, adj):
        aggregate = torch.matmul(adj, input)
        output = torch.cat([input, aggregate], dim=2)
        output = self.fc(output)

        output = self.relu(output)

        return output


