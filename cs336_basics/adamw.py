"""AdamW optimizer (Loshchilov & Hutter, 2019) with decoupled weight decay."""

import math

import torch
from torch.optim.optimizer import Optimizer


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate :{lr}")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            with torch.no_grad():
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    # per-parameter state: step count t, first moment m, second moment v
                    state = self.state[p]
                    t = state.get("t", 1)
                    m = state.get("m", 0)
                    v = state.get("v", 0)
                    # bias-corrected step size
                    alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                    grad = p.grad
                    # decoupled weight decay
                    p -= lr * weight_decay * p
                    state["m"] = beta1 * m + (1 - beta1) * grad
                    state["v"] = beta2 * v + (1 - beta2) * grad**2
                    # use the most up-to-date m and v to update the params
                    p -= alpha_t * state["m"] / (torch.sqrt(state["v"]) + eps)
                    state["t"] = t + 1
        return loss
