import torch
from torch import nn
import numpy as np

from layers.baselayer import Layer


class ReLuLayer(Layer):

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, c=6, initialization=None):
        super().__init__(in_features, out_features, bias, is_first, omega_0, c)

        self.relu = nn.ReLU()

        if initialization is None:
            with torch.no_grad():
                if self.is_first:
                    self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
                else:
                    self.linear.weight.uniform_(-np.sqrt(self.c / self.in_features) / self.omega_0,
                                                np.sqrt(self.c / self.in_features) / self.omega_0)

        else:
            initialization(self.linear.weight)

    def forward(self, input):
        return self.relu(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.relu(intermediate), intermediate
