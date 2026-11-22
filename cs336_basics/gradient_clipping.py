"""Global L2-norm gradient clipping (in place)."""

import math
from collections.abc import Iterable

import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    parameters = list(parameters)
    with torch.no_grad():
        # global norm over all parameter gradients
        cumulative_l2_sq = 0
        for param in parameters:
            g = param.grad
            if g is None:
                continue
            l2_norm = torch.norm(g, p=2)
            cumulative_l2_sq += l2_norm**2
        cumulative_l2 = math.sqrt(cumulative_l2_sq)
        # rescale every gradient by the same factor if the global norm is too large
        if cumulative_l2 >= max_l2_norm:
            scale = max_l2_norm / (cumulative_l2 + eps)
            for param in parameters:
                g = param.grad
                if g is None:
                    continue
                param.grad *= scale
