"""Numerically stable softmax."""

import torch


def softmax(x: torch.Tensor, dim: int):
    max_x = torch.max(x, dim=dim, keepdim=True).values
    exps = torch.exp(x - max_x)
    denom = torch.sum(exps, dim=dim, keepdim=True)
    return exps / denom
