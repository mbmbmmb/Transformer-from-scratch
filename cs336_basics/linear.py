"""Bias-free linear layer, y = x W^T."""

import math

import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, weights=None, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        if weights is None:
            # N(0, 2 / (d_in + d_out)) truncated to [-3 sigma, 3 sigma]
            sigma = math.sqrt(2 / (in_features + out_features))
            linear_init = torch.normal(mean=0, std=sigma, size=(out_features, in_features))
            linear_init = linear_init.clip(-3 * sigma, 3 * sigma)
            self.W = nn.Parameter(linear_init, requires_grad=True)
        else:
            self.W = nn.Parameter(weights, requires_grad=True)

    def forward(self, x: torch.Tensor):
        # x: (..., d_in) -> (..., d_out)
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")
