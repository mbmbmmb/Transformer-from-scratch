"""Cosine learning-rate schedule with linear warmup."""

import math


def cos_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < 0:
        raise ValueError(f"Iteration step must be non-negative, input: {it}")
    elif it < warmup_iters:
        # linear warmup from 0
        return it / warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        # cosine decay from max to min
        return min_learning_rate + 0.5 * (
            1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)
        ) * (max_learning_rate - min_learning_rate)
    else:
        # constant at min after the cycle
        return min_learning_rate
