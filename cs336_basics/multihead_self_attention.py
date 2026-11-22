"""Causal multi-head self-attention, without and with RoPE."""

import math

import torch
import torch.nn as nn

from .rope import RoPE


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention without positional information."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        o_proj_weight=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        sig = math.sqrt(1 / d_model)
        if q_proj_weight is not None:
            self.q_proj_weight = q_proj_weight
        else:
            self.q_proj_weight = torch.normal(mean=0, std=sig, size=(d_model, d_model)).clip(-sig, sig)
        if k_proj_weight is not None:
            self.k_proj_weight = k_proj_weight
        else:
            self.k_proj_weight = torch.normal(mean=0, std=sig, size=(d_model, d_model)).clip(-sig, sig)
        if v_proj_weight is not None:
            self.v_proj_weight = v_proj_weight
        else:
            self.v_proj_weight = torch.normal(mean=0, std=sig, size=(d_model, d_model)).clip(-sig, sig)
        if o_proj_weight is not None:
            self.o_proj_weight = o_proj_weight
        else:
            self.o_proj_weight = torch.normal(mean=0, std=sig, size=(d_model, d_model)).clip(-sig, sig)
        self.q_proj_weight = nn.Parameter(self.q_proj_weight, requires_grad=True)
        self.k_proj_weight = nn.Parameter(self.k_proj_weight, requires_grad=True)
        self.v_proj_weight = nn.Parameter(self.v_proj_weight, requires_grad=True)
        self.o_proj_weight = nn.Parameter(self.o_proj_weight, requires_grad=True)

    def forward(self, x):
        # x: [batch, ..., seq, d_model]
        Q = torch.matmul(x, self.q_proj_weight.T)
        K = torch.matmul(x, self.k_proj_weight.T)
        V = torch.matmul(x, self.v_proj_weight.T)
        # split heads: [batch, ..., seq, d_model] -> [batch, ..., num_heads, seq, d_k]
        merge_shape = Q.shape
        Q = Q.reshape(*merge_shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)
        K = K.reshape(*merge_shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)
        V = V.reshape(*merge_shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)

        # scores: [batch, ..., num_heads, seq, seq]
        scale_dot = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        seq = scale_dot.shape[-1]
        # causal mask: True above the diagonal = future positions, filled with -inf
        masks = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1)
        scale_dot_masked = scale_dot.masked_fill(masks, -float("inf"))
        scale_wt_causal = torch.softmax(scale_dot_masked, dim=-1)
        att_wt_causal = torch.matmul(scale_wt_causal, V)
        # merge heads: [batch, ..., num_heads, seq, d_k] -> [batch, ..., seq, d_model]
        att_wt_causal = att_wt_causal.transpose(-3, -2)
        att_wt_causal = att_wt_causal.reshape(*merge_shape)

        return torch.matmul(att_wt_causal, self.o_proj_weight.T)


class MultiHeadSelfAttentionWithRoPE(nn.Module):
    """Causal multi-head self-attention with RoPE applied to Q and K.

    `no_rope=True` skips the rotation (NoPE ablation); the RoPE tables are still built.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta,
        max_seq_len,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        o_proj_weight=None,
        no_rope=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.no_rope = no_rope
        self.rope = RoPE(theta, self.d_k, max_seq_len)
        sig = math.sqrt(1 / d_model)
        if q_proj_weight is not None:
            self.q_proj_weight = q_proj_weight
        else:
            self.q_proj_weight = torch.normal(mean=0, std=sig, size=(d_model, d_model)).clip(-3 * sig, 3 * sig)
        if k_proj_weight is not None:
            self.k_proj_weight = k_proj_weight
        else:
            self.k_proj_weight = torch.normal(mean=0, std=sig, size=(d_model, d_model)).clip(-3 * sig, 3 * sig)
        if v_proj_weight is not None:
            self.v_proj_weight = v_proj_weight
        else:
            self.v_proj_weight = torch.normal(mean=0, std=sig, size=(d_model, d_model)).clip(-3 * sig, 3 * sig)
        if o_proj_weight is not None:
            self.o_proj_weight = o_proj_weight
        else:
            self.o_proj_weight = torch.normal(mean=0, std=sig, size=(d_model, d_model)).clip(-3 * sig, 3 * sig)
        self.q_proj_weight = nn.Parameter(self.q_proj_weight, requires_grad=True)
        self.k_proj_weight = nn.Parameter(self.k_proj_weight, requires_grad=True)
        self.v_proj_weight = nn.Parameter(self.v_proj_weight, requires_grad=True)
        self.o_proj_weight = nn.Parameter(self.o_proj_weight, requires_grad=True)

    def forward(self, x, token_positions=None):
        # x: [batch, ..., seq, d_model]
        if token_positions is None:
            token_positions = torch.arange(0, x.shape[-2], step=1, device=x.device)
        Q = torch.matmul(x, self.q_proj_weight.T)
        K = torch.matmul(x, self.k_proj_weight.T)
        V = torch.matmul(x, self.v_proj_weight.T)
        # split heads: [batch, ..., seq, d_model] -> [batch, ..., num_heads, seq, d_k]
        merge_shape = Q.shape
        Q = Q.reshape(*merge_shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)
        K = K.reshape(*merge_shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)
        # RoPE is applied per head to Q and K only
        if not self.no_rope:
            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)
        V = V.reshape(*merge_shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)

        # scores: [batch, ..., num_heads, seq, seq]
        scale_dot = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        seq = scale_dot.shape[-1]
        # causal mask: True above the diagonal = future positions, filled with -inf
        masks = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1)
        scale_dot_masked = scale_dot.masked_fill(masks, -float("inf"))
        scale_wt_causal = torch.softmax(scale_dot_masked, dim=-1)
        att_wt_causal = torch.matmul(scale_wt_causal, V)
        # merge heads: [batch, ..., num_heads, seq, d_k] -> [batch, ..., seq, d_model]
        att_wt_causal = att_wt_causal.transpose(-3, -2)
        att_wt_causal = att_wt_causal.reshape(*merge_shape)

        return torch.matmul(att_wt_causal, self.o_proj_weight.T)
