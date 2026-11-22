#!/usr/bin/env bash
# layer_norm_ablation — remove all 9 RMSNorms, at the optimal LR and lower.
set -u
cd "$(dirname "$0")/.."
ROOT=runs/base/ablations
run () { name=$1; iters=$2; shift 2
  out="$ROOT/$name"
  if compgen -G "$out/*/metrics.jsonl" >/dev/null; then echo "== skip $name"; return; fi
  echo "== $name"
  uv run python -u -m cs336_basics.train --override \
    training.max_iters=$iters training.eval_interval=50 training.checkpoint_interval=999999 \
    model.no_rmsnorm=True logging.out_dir="$out" "$@" 2>&1 | tail -2
}
lr () { run "$1" "$2" optim.lr=$3 lr_schedule.max_learning_rate=$3 lr_schedule.min_learning_rate=$4; }
lr nonorm_lr1e-3  600 1.0e-3 1.0e-4     # previous optimal LR — expect blowup
lr nonorm_lr1e-4 2000 1.0e-4 1.0e-5
lr nonorm_lr1e-5 2000 1.0e-5 1.0e-6
echo "NORM ABLATION DONE"
