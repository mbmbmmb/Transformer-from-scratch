"""Rotary position embeddings (RoPE) with precomputed cos/sin tables."""

import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        # angle table theta_{i,k} = i * theta^(-2k / d_k), shape (max_seq_len, d_k // 2)
        pos = torch.arange(0, max_seq_len, 1, dtype=torch.float32, device=device).reshape((max_seq_len, 1))
        dims = (theta ** (-torch.arange(0, d_k, 2, device=device) / d_k)).reshape((1, d_k // 2))
        theta_matrix = pos @ dims

        cos = torch.cos(theta_matrix)
        sin = torch.sin(theta_matrix)
        # buffers move with model.to(device) but are not trained
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x, token_positions):
        # x: (..., seq_len, d_k), token_positions: (..., seq_len)
        cos_a = self.cos[token_positions, :]
        sin_a = self.sin[token_positions, :]
        # rotate each (even, odd) pair of features, then interleave back
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], dim=-1).flatten(-2)
