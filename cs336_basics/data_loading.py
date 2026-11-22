"""Random batch sampling from a 1-D token-id array (works with np.memmap)."""

import numpy as np
import torch


def data_loading(dataset: np.ndarray, batch_size: int, context_length: int, device: str, rng=None):
    """
    Sample `batch_size` random windows from the token stream.

    For a start index i, the input is dataset[i : i + context_length] and the label is the
    same window shifted by one, dataset[i + 1 : i + context_length + 1] (next-token targets).
    Returns (inputs, labels), each of shape (batch_size, context_length), dtype long.

    `dataset` can be an np.memmap: fancy indexing only pages in the windows it touches,
    so the full corpus never has to fit in RAM. Pass `rng` (e.g. np.random.RandomState)
    for reproducible batches, e.g. a fixed set of validation batches.
    """
    r = rng if rng is not None else np.random
    # start indices in [0, len - context_length), broadcast to (batch_size, context_length)
    input_idx = r.randint(low=0, high=len(dataset) - context_length, size=(batch_size, 1)) + np.arange(context_length)
    label_idx = input_idx + 1
    input = torch.tensor(dataset[input_idx], device=device, dtype=torch.long)
    label = torch.tensor(dataset[label_idx], device=device, dtype=torch.long)
    return input, label
