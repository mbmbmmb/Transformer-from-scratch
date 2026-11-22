"""Save / restore model + optimizer state and the iteration count."""

import os
from typing import IO, BinaryIO

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint_dic = {}
    checkpoint_dic["model_state"] = model.state_dict()
    checkpoint_dic["optimizer_state"] = optimizer.state_dict()
    checkpoint_dic["iteration"] = iteration
    torch.save(checkpoint_dic, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
):
    checkpoint_dic = torch.load(src)
    model.load_state_dict(checkpoint_dic["model_state"])
    optimizer.load_state_dict(checkpoint_dic["optimizer_state"])
    return checkpoint_dic["iteration"]
