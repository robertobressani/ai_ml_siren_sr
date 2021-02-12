from collections import OrderedDict

import torch
import numpy as np
from torch import nn
from layers.sinelayer import SineLayer
from layers.relulayer import ReLuLayer


class NetworkDimensions:
    def __init__(self, in_features, out_features, hidden_features=256, hidden_layers=2, first_omega_0=30, hidden_omega_0=30.,
                 last_omega_0=30.):
        self.out_features = out_features
        self.hidden_layers = hidden_layers
        self.hidden_features = hidden_features
        self.in_features = in_features
        self.hidden_omega_0 = hidden_omega_0
        self.first_omega_0 = first_omega_0
        self.last_omega_0 = last_omega_0


class NetworkParams:
    def __init__(self, layer_class=SineLayer, outermost_linear=True, c=6, description='sine_layer', first_init=None, hidden_init=None):
        self.c = c
        self.outermost_linear = outermost_linear
        self.layer_class = layer_class
        self.description = description
        self.first_init = first_init
        self.hidden_init = hidden_init


class Network(nn.Module):
    def __init__(self, params: NetworkParams, dimensions: NetworkDimensions):
        super().__init__()

        self.net = []
        self.net.append(params.layer_class(dimensions.in_features, dimensions.hidden_features,
                                           is_first=True, omega_0=dimensions.first_omega_0, c=params.c,
                                           initialization=params.first_init))

        for i in range(dimensions.hidden_layers):
            self.net.append(params.layer_class(dimensions.hidden_features, dimensions.hidden_features,
                                               is_first=False, omega_0=dimensions.hidden_omega_0, c=params.c,
                                               initialization=params.hidden_init))

        if params.outermost_linear:
            final_linear = nn.Linear(dimensions.hidden_features, dimensions.out_features)

            with torch.no_grad():
                if params.layer_class == SineLayer:
                    final_linear.weight.uniform_(
                        -np.sqrt(params.c / dimensions.hidden_features) / dimensions.hidden_omega_0,
                        np.sqrt(params.c / dimensions.hidden_features) / dimensions.hidden_omega_0)

            self.net[-1].omega_0 = dimensions.last_omega_0
            self.net.append(final_linear)
        else:
            self.net.append(params.layer_class(dimensions.hidden_features, dimensions.out_features,
                                               is_first=False, omega_0=dimensions.last_omega_0, c=params.c))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer) or isinstance(layer, ReLuLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
