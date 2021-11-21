import RBFnet.RBF as rbf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np


class Network(torch.nn.Module):

    def __init__(self, layer_widths, layer_centres, basis_func):
        super(Network, self).__init__()
        self.model = nn.ModuleList()
        self.model.append(rbf.RBF(layer_widths, layer_centres, basis_func))
        self.model.append(nn.Linear(layer_centres,1))

    def forward(self, x):
        y = self.model[0](x)
        out = self.model[1](y)
        return out
