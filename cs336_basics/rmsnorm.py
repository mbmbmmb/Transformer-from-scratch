"""Root-mean-square layer normalization with a learned gain."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, weights=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        if weights is None:
            self.g = nn.Parameter(torch.ones(size=(d_model,)))
        else:
            self.g = nn.Parameter(weights)

    def forward(self, x: torch.Tensor):
        # x: (batch, seq, d_model); normalize in float32, then cast back
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        normed = (x * self.g) / rms
        return normed.to(in_dtype)
