#!/usr/bin/env bash
# swiglu_ablation — ungated W2.SiLU(W1 x) with d_ff = 4*d_model to match params.
# 2000 steps at lr 1e-3, comparable to runs/base/ablations/lr1e-3 (val 1.988).
set -u
cd "$(dirname "$0")/.."
out=runs/base/ablations/silu
if compgen -G "$out/*/metrics.jsonl" >/dev/null; then echo "== skip silu"; else
  uv run python -u -m cs336_basics.train --override \
    training.max_iters=2000 training.eval_interval=50 training.checkpoint_interval=999999 \
    model.silu=True model.d_ff=2048 logging.out_dir="$out" 2>&1 | tail -2
fi
echo "SILU DONE"
