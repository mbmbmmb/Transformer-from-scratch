#!/usr/bin/env bash
# no_pos_emb — RoPE removed from Q and K. 2000 steps at lr 1e-3,
# directly comparable to runs/base/ablations/lr1e-3 (val 1.988).
set -u
cd "$(dirname "$0")/.."
out=runs/base/ablations/nope
if compgen -G "$out/*/metrics.jsonl" >/dev/null; then echo "== skip nope"; else
  uv run python -u -m cs336_basics.train --override \
    training.max_iters=2000 training.eval_interval=50 training.checkpoint_interval=999999 \
    model.no_rope=True logging.out_dir="$out" 2>&1 | tail -2
fi
echo "NOPE DONE"
