#!/usr/bin/env bash
# Run a queue of cold-start training runs on one GPU, then evaluate them all.
#   CUDA_VISIBLE_DEVICES=1 scripts/run_queue.sh <out_dir> recipe:seed:budget_min [...]
# Each GPU gets its own compilation cache root, so two queues on one machine never
# share or wipe each other's kernel caches.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$1"; shift
mkdir -p "$out"
export RLD2_CACHE_ROOT="${RLD2_CACHE_ROOT:-/root/.rld2_cache/gpu${CUDA_VISIBLE_DEVICES:-0}}"
for item in "$@"; do
  IFS=: read -r recipe seed budget <<< "$item"
  run="$out/${recipe}_s${seed}"
  [[ -f "$run/run.json" ]] && grep -q finished_ts "$run/run.json" && continue
  rm -rf "$run"
  RLD2_COLD=1 "$here/scripts/train.sh" "$recipe" "$seed" "$run" "$budget" > "$run.log" 2>&1
done
for item in "$@"; do
  IFS=: read -r recipe seed budget <<< "$item"
  run="$out/${recipe}_s${seed}"
  [[ -f "$run/time_to_bar.json" ]] || MUJOCO_GL=egl "$here/.venv/bin/python" -m rl_d2.eval.time_to_bar \
    --run "$run" --procs "${RLD2_EVAL_PROCS:-6}" > "$run.eval.log" 2>&1
done
