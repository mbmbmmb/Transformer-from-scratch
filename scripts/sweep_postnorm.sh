#!/usr/bin/env bash
# pre_norm_ablation — post-norm blocks vs the pre-norm baseline.
# 2000 steps at lr 1e-3 makes these directly comparable to runs/base/ablations/lr1e-3.
set -u
cd "$(dirname "$0")/.."
ROOT=runs/base/ablations
run () { name=$1; shift
  out="$ROOT/$name"
  if compgen -G "$out/*/metrics.jsonl" >/dev/null; then echo "== skip $name"; return; fi
  echo "== $name"
  uv run python -u -m cs336_basics.train --override \
    training.max_iters=2000 training.eval_interval=50 training.checkpoint_interval=999999 \
    model.post_norm=True logging.out_dir="$out" "$@" 2>&1 | tail -2
}
run postnorm_lr1e-3 optim.lr=1.0e-3 lr_schedule.max_learning_rate=1.0e-3 lr_schedule.min_learning_rate=1.0e-4
run postnorm_lr3e-4 optim.lr=3.0e-4 lr_schedule.max_learning_rate=3.0e-4 lr_schedule.min_learning_rate=3.0e-5
echo "POSTNORM DONE"
