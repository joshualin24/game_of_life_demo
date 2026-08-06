"""
Minimal CNN architecture for verifying arXiv:2606.23587 (Ahmed & Davis):
"It's Much Easier for Neural Networks to Learn Game of Life Dynamics with
the Right Activation Function: Polynomial Kolmogorov-Arnold Networks."

Architecture follows the paper's L(n=1, m) design:
  Conv2d(1, 2m, kernel=3, circular padding)   -- local Moore-neighborhood features
  activation
  Conv2d(2m, m, kernel=1)                     -- 1x1 channel mixing
  activation
  Conv2d(m, 1, kernel=1)                      -- output logit per cell (sigmoid applied externally)

make_relu_net() and make_poly_net() build the *same* MinimalCA class and
differ ONLY in which activation module is placed at the two hidden
activation spots.
"""

import torch
import torch.nn as nn


class PolyActivation(nn.Module):
    """Learnable per-channel 2nd-degree polynomial: phi(x) = w0 + w1*x + w2*x^2."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.w1 = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.w2 = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w0 + self.w1 * x + self.w2 * x * x


class MinimalCA(nn.Module):
    """
    L(1, m) architecture. `activation_factory(num_channels)` builds the
    activation module for a given layer width, e.g. `lambda c: nn.ReLU()`
    or `lambda c: PolyActivation(c)`.
    """

    def __init__(self, activation_factory, m: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2 * m, kernel_size=3, padding=1,
                                padding_mode='circular', bias=True)
        self.act1  = activation_factory(2 * m)
        self.conv2 = nn.Conv2d(2 * m, m, kernel_size=1, bias=True)
        self.act2  = activation_factory(m)
        self.conv3 = nn.Conv2d(m, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return self.conv3(x)   # logits; apply sigmoid externally


def make_relu_net(m: int = 1) -> MinimalCA:
    return MinimalCA(lambda c: nn.ReLU(), m=m)


def make_poly_net(m: int = 1) -> MinimalCA:
    return MinimalCA(lambda c: PolyActivation(c), m=m)
