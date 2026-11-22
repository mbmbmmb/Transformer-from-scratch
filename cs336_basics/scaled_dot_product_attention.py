"""Scaled dot-product attention with an optional boolean mask."""

import math

import torch

from .softmax import softmax


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K: (batch, ..., seq_len, d_k); V: (batch, ..., seq_len, d_v)
    mask: boolean, True = position may be attended to.
    Returns (batch, ..., seq_len, d_v).
    """
    d_k = Q.shape[-1]
    scale_dot_prod = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # mask is True where attention is allowed, so fill the complement with -inf
        scale_dot_prod = scale_dot_prod.masked_fill(mask=~mask, value=-float("inf"))
    dot_prod_wts = softmax(scale_dot_prod, dim=-1)
    return torch.matmul(dot_prod_wts, V)
