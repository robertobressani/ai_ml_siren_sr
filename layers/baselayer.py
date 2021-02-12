from torch import nn


class Layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, c=6):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.c = c
        self.linear = nn.Linear(in_features, out_features, bias=bias)
