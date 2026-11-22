"""Train the Transformer LM on a pre-tokenized uint16 .bin corpus.

    uv run python -m cs336_basics.train --config configs/base.yaml \
        --override optim.lr=3e-4 lr_schedule.max_learning_rate=3e-4

Every `eval_interval` steps appends {t, train_loss, test_loss (validation), lr, elapsed, tokens}
to <out_dir>/<run_name>/metrics.jsonl; saves a checkpoint every `checkpoint_interval` steps.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import save_checkpoint
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import data_loading
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.learning_rate_schedule import cos_schedule
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.utils import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def evaluate(model, valid_data, cfg):
    """Mean validation loss over `eval_iters` batches. A private RNG with a fixed seed gives
    the same validation batches at every eval and leaves the training RNG untouched."""
    rng = np.random.RandomState(cfg["training"]["seed"])
    eval_loss = []
    model.eval()
    with torch.no_grad():
        for _ in range(cfg["training"]["eval_iters"]):
            input, label = data_loading(valid_data, **cfg["data"], rng=rng)
            logits = model.forward(input)
            loss = cross_entropy(logits.reshape((-1, logits.shape[-1])), label.reshape(-1))
            eval_loss.append(loss.item())
    model.train()
    return sum(eval_loss) / len(eval_loss)


def main(cfg):
    np.random.seed(cfg["training"]["seed"])
    torch.manual_seed(cfg["training"]["seed"])
    # memmap: token ids are paged in on demand, the corpus never has to fit in RAM
    train_data = np.memmap(cfg["paths"]["train_bin"], dtype=np.uint16, mode="r")
    valid_data = np.memmap(cfg["paths"]["valid_bin"], dtype=np.uint16, mode="r")

    tsm_lm = TransformerLM(**cfg["model"])
    opt = AdamW(tsm_lm.parameters(), **cfg["optim"])
    tsm_lm.to(device=cfg["training"]["device"])

    max_iters = cfg["training"]["max_iters"]
    run_name = cfg["logging"]["run_name"] or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = cfg["logging"]["out_dir"] / run_name
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.perf_counter()
    for t in range(max_iters):
        lr = cos_schedule(it=t, **cfg["lr_schedule"])
        input, label = data_loading(train_data, **cfg["data"])

        logits = tsm_lm.forward(input)
        loss = cross_entropy(logits.reshape((-1, logits.shape[-1])), label.reshape(-1))

        # zero_grad  ──before──>  backward   (else grads accumulate across steps)
        opt.zero_grad()
        # backward   ──before──>  clip       (clip needs .grad populated)
        loss.backward()
        # clip       ──before──>  step       (else you step on unclipped grads)
        gradient_clipping(tsm_lm.parameters(), **cfg["gradient_clipping"])
        # lr write   ──before──>  step       (step reads group["lr"])
        opt.param_groups[0]["lr"] = lr
        opt.step()

        if (t + 1) % cfg["training"]["eval_interval"] == 0:
            print("Iter:", t, " loss: ", loss.item())
            elapsed = time.perf_counter() - t_start
            train_loss = loss.item()
            test_loss = evaluate(tsm_lm, valid_data, cfg)
            record = {
                "t": t,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "lr": lr,
                "elapsed": round(elapsed, 1),
                "tokens": (t + 1) * cfg["data"]["batch_size"] * cfg["data"]["context_length"],
            }
            with open(out_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
        if (t + 1) % cfg["training"]["checkpoint_interval"] == 0:
            save_checkpoint(model=tsm_lm, optimizer=opt, iteration=t, out=out_dir / f"ckpt_{t}.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--override", nargs="*", default=[], dest="overrides")
    args = p.parse_args()

    config_path = REPO_ROOT / args.config
    cfg = load_config(REPO_ROOT, config_path, args.overrides)
    main(cfg)
