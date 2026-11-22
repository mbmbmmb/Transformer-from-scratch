"""Token embedding lookup."""

import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, weights=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        if weights is None:
            # N(0, 1) truncated to [-3, 3]
            init_param = torch.normal(mean=0, std=1, size=(num_embeddings, embedding_dim)).clip(-3, 3)
            self.weights = nn.Parameter(init_param)
        else:
            self.weights = nn.Parameter(weights)

    def forward(self, token_ids):
        # token_ids: (batch, seq_len) -> (batch, seq_len, embedding_dim)
        return self.weights[token_ids]
