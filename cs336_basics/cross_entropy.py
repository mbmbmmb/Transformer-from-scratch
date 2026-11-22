"""Numerically stable cross-entropy (log-sum-exp with max subtraction)."""

import torch


def cross_entropy(inputs, targets):
    """
    inputs: [batch_size, vocab_size] unnormalized logits
    targets: [batch_size], each element is a class index in [0, vocab_size)
    Returns the mean negative log-likelihood over the batch.
    """
    batch, _ = inputs.shape
    inputs_xi = inputs[torch.arange(batch, device=inputs.device), targets]
    # subtract the row max before exp() for numerical stability
    M = torch.max(inputs, dim=-1, keepdim=True).values
    M_flat = torch.max(inputs, dim=-1).values
    return -(inputs_xi - M_flat).sum() / batch + torch.log(torch.exp(inputs - M).sum(dim=-1)).sum() / batch
