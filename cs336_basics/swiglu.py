"""SwiGLU feed-forward network: W2 (SiLU(W1 x) * W3 x).

With `silu=True` it becomes the ungated SiLU FFN, W2 SiLU(W1 x), used for the
SwiGLU-vs-SiLU ablation (W3 is then not created).
"""

import math

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


def silu(x):
    return x * torch.sigmoid(x)


# d_ff is taken as given; the config uses d_ff ~ 8/3 * d_model rounded to a multiple of 64.
class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        w1_weight: Float[Tensor, " d_ff d_model"] | None,
        w2_weight: Float[Tensor, " d_model d_ff"] | None,
        w3_weight: Float[Tensor, " d_ff d_model"] | None,
        silu=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.silu = silu
        std_ffn = 2 / math.sqrt(self.d_ff + self.d_model)

        if w1_weight is None:
            self.W1 = nn.Parameter(
                torch.normal(size=(self.d_ff, self.d_model), mean=0, std=std_ffn).clip(-3 * std_ffn, 3 * std_ffn)
            )
        else:
            self.W1 = nn.Parameter(w1_weight)
        if w2_weight is None:
            self.W2 = nn.Parameter(
                torch.normal(size=(self.d_model, self.d_ff), mean=0, std=std_ffn).clip(-3 * std_ffn, 3 * std_ffn)
            )
        else:
            self.W2 = nn.Parameter(w2_weight)
        if not silu:
            if w3_weight is None:
                self.W3 = nn.Parameter(
                    torch.normal(size=(self.d_ff, self.d_model), mean=0, std=std_ffn).clip(-3 * std_ffn, 3 * std_ffn)
                )
            else:
                self.W3 = nn.Parameter(w3_weight)

    def forward(self, x):
        # x: (..., d_model); W1, W3: (d_ff, d_model); W2: (d_model, d_ff)
        if self.silu:
            return torch.matmul(silu(torch.matmul(x, self.W1.T)), self.W2.T)
        else:
            return torch.matmul((silu(torch.matmul(x, self.W1.T)) * (torch.matmul(x, self.W3.T))), self.W2.T)
