"""One Transformer decoder block (pre-norm by default).

Ablation switches:
  no_rmsnorm -> x + attn(x); x + ffn(x)               (no normalization at all)
  post_norm  -> norm(x + attn(x)); norm(x + ffn(x))
  no_rope    -> attention without RoPE (NoPE)
  silu       -> ungated SiLU FFN instead of SwiGLU
"""

import torch
import torch.nn as nn
from torch import Tensor

from .multihead_self_attention import MultiHeadSelfAttentionWithRoPE
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, Tensor] | None,
        no_rmsnorm=False,
        post_norm=False,
        no_rope=False,
        silu=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        # ablation switches
        self.no_rmsnorm = no_rmsnorm
        self.post_norm = post_norm
        self.no_rope = no_rope
        self.silu = silu
        if weights is None:
            # None = each sub-module uses its own random init
            self.weights = {}
            # RMSNorm weights -> 1, shape (d_model, )
            self.weights["ln1.weight"] = None
            self.weights["ln2.weight"] = None
            # Linear Layer, projection matrices Q, K, V, output
            # they all have shape (d_model, d_model)
            self.weights["attn.q_proj.weight"] = None
            self.weights["attn.k_proj.weight"] = None
            self.weights["attn.v_proj.weight"] = None
            self.weights["attn.output_proj.weight"] = None
            # FFN: w1 (d_ff, d_model), w2 (d_model, d_ff), w3 (d_ff, d_model)
            self.weights["ffn.w1.weight"] = None
            self.weights["ffn.w2.weight"] = None
            self.weights["ffn.w3.weight"] = None
        else:
            self.weights = weights

        self.rmsnorm1 = RMSNorm(d_model=self.d_model, weights=self.weights["ln1.weight"])
        # causal multi-head attention
        self.att = MultiHeadSelfAttentionWithRoPE(
            d_model=self.d_model,
            num_heads=self.num_heads,
            theta=self.theta,
            max_seq_len=self.max_seq_len,
            q_proj_weight=self.weights["attn.q_proj.weight"],
            k_proj_weight=self.weights["attn.k_proj.weight"],
            v_proj_weight=self.weights["attn.v_proj.weight"],
            o_proj_weight=self.weights["attn.output_proj.weight"],
            no_rope=no_rope,
        )
        self.rmsnorm2 = RMSNorm(d_model=self.d_model, weights=self.weights["ln2.weight"])
        self.ffn = SwiGLU(
            d_model=self.d_model,
            d_ff=self.d_ff,
            w1_weight=self.weights["ffn.w1.weight"],
            w2_weight=self.weights["ffn.w2.weight"],
            w3_weight=self.weights["ffn.w3.weight"],
            silu=silu,
        )

    def forward(self, x, token_positions=None):
        if token_positions is None:
            token_positions = torch.arange(0, x.shape[-2], step=1, device=x.device)
        if self.no_rmsnorm:
            x = x + self.att.forward(x, token_positions)
            x = x + self.ffn.forward(x)
        elif self.post_norm:
            x = x + self.att.forward(x, token_positions)
            x = self.rmsnorm1(x)
            x = x + self.ffn(x)
            x = self.rmsnorm2(x)
        else:
            # pre-norm (default)
            x = x + self.att.forward(self.rmsnorm1.forward(x), token_positions)
            x = x + self.ffn.forward(self.rmsnorm2(x))
        return x
