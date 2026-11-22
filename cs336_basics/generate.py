"""Sample text from a trained checkpoint with temperature sampling.

    uv run python -m cs336_basics.generate --config configs/base.yaml --prompt "Once upon a time"

Loads <logging.out_dir>/ckpt_{max_iters - 1}.pt and the tokenizer from paths.vocab / paths.merges.
"""

import argparse
from pathlib import Path

import torch

from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import load_checkpoint
from cs336_basics.train_bpe import Tokenizer
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.utils import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def generate_text(prompt, output_length, temperature, cfg):
    vocab_path = cfg["paths"]["vocab"]
    merges_path = cfg["paths"]["merges"]

    # the last checkpoint written by train.py
    ckpt_path = cfg["logging"]["out_dir"] / f"ckpt_{cfg['training']['max_iters'] - 1}.pt"
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=cfg["tokenizer"]["special_tokens"],
    )

    model = TransformerLM(**cfg["model"])
    opt = AdamW(params=list(model.parameters()))
    load_checkpoint(ckpt_path, model, opt)
    ids = tokenizer.encode(prompt)
    ids = torch.tensor([ids], device=cfg["training"]["device"], dtype=torch.long)
    model.to(ids.device)
    start_len = ids.shape[-1]

    # autoregressive sampling: forward the whole sequence, sample the next id from the last position
    while ids.shape[-1] - start_len < output_length:
        logits = model(ids)
        probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
        pred = torch.multinomial(probs, 1)
        ids = torch.cat([ids, pred], dim=1)
    return tokenizer.decode(ids[0].tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--output_length", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()

    config_path = REPO_ROOT / args.config
    cfg = load_config(REPO_ROOT, config_path)

    output_text = generate_text(
        prompt=args.prompt, output_length=args.output_length, temperature=args.temperature, cfg=cfg
    )
    print(output_text)
