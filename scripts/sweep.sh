#!/usr/bin/env bash
# Ablation sweeps for CS336 A1 §7.
#   batch_size_experiment : vary data.batch_size
#   learning_rate         : vary the LR schedule (incl. a divergent run)
# Idempotent: a config whose output already exists is skipped, so an
# interrupted sweep can be resumed by re-running this script.
set -u
cd "$(dirname "$0")/.."

COMMON="training.max_iters=2000 training.eval_interval=100 training.checkpoint_interval=999999"
ROOT=runs/base/ablations
mkdir -p "$ROOT"

run () {                        # run <name> <override>...
  local name=$1; shift
  local out="$ROOT/$name"
  # run_name is unset -> main() creates <out>/<timestamp>/, so glob for it
  if compgen -G "$out/*/metrics.jsonl" > /dev/null; then
    echo "== skip $name (already has results)"; return
  fi
  echo "== $name : $*"
  uv run python -u -m cs336_basics.train \
      --override $COMMON logging.out_dir="$out" "$@" 2>&1 | tee -a "$ROOT/sweep.log"
}

# --- batch_size_experiment (128 last: 17.5 GiB, ~60 min) ---
for B in 1 16 32 64; do
  run "batch$B" data.batch_size=$B
done

# --- learning_rate sweep; all three keys must move together, because
#     the schedule overwrites param_groups[0]["lr"] every step ---
lr_run () {                     # lr_run <name> <max_lr> <min_lr>
  run "$1" optim.lr=$2 lr_schedule.max_learning_rate=$2 lr_schedule.min_learning_rate=$3
}
lr_run lr1e-4 1.0e-4 1.0e-5
lr_run lr3e-4 3.0e-4 3.0e-5
lr_run lr1e-3 1.0e-3 1.0e-4     # baseline
lr_run lr3e-3 3.0e-3 3.0e-4
lr_run lr1e-2 1.0e-2 1.0e-3     # expected to diverge (edge of stability)

echo "ALL DONE"
