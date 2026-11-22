"""Decoder-only Transformer language model.

token ids -> Embedding -> num_layers x TransformerBlock -> RMSNorm -> Linear (LM head) -> logits
"""

import torch.nn as nn
from torch import Tensor

from .embedding import Embedding
from .linear import Linear
from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: int,
        weights: dict[str, Tensor] | None = None,
        no_rmsnorm=False,
        post_norm=False,
        no_rope=False,
        silu=False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.rope_theta = rope_theta
        self.weights = weights

        # ablation switches
        self.no_rmsnorm = no_rmsnorm
        self.post_norm = post_norm
        self.no_rope = no_rope

        self.embed = Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model,
            weights=self.weights["token_embeddings.weight"] if weights else None,
        )
        # nn.ModuleList (not a plain list) so the blocks are registered as sub-modules
        self.transform_layers = nn.ModuleList()
        for i in range(self.num_layers):
            # pick out this layer's weights from the flat state dict, if given
            layer_weights = {}
            for name in ["q_proj", "k_proj", "v_proj", "output_proj"]:
                layer_weights[f"attn.{name}.weight"] = (
                    self.weights[f"layers.{i}.attn.{name}.weight"] if weights else None
                )
            for name in ["ln1", "ffn.w1", "ffn.w2", "ffn.w3", "ln2"]:
                layer_weights[f"{name}.weight"] = self.weights[f"layers.{i}.{name}.weight"] if weights else None

            transform_layer = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                max_seq_len=self.context_length,
                theta=self.rope_theta,
                weights=layer_weights,
                no_rmsnorm=no_rmsnorm,
                post_norm=post_norm,
                no_rope=no_rope,
                silu=silu,
            )
            self.transform_layers.append(transform_layer)
        if not no_rmsnorm:
            self.rmsnorm = RMSNorm(
                d_model=self.d_model,
                weights=self.weights["ln_final.weight"] if weights else None,
            )
        # LM head
        self.linear = Linear(
            in_features=self.d_model,
            out_features=self.vocab_size,
            weights=self.weights["lm_head.weight"] if weights else None,
        )

    def forward(self, in_indices):
        # in_indices: (batch, seq) -> logits (batch, seq, vocab_size)
        x = self.embed.forward(token_ids=in_indices)
        for i in range(self.num_layers):
            x = self.transform_layers[i].forward(x)
        if not self.no_rmsnorm:
            x = self.rmsnorm.forward(x)
        x = self.linear.forward(x)
        return x
