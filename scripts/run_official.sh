#!/usr/bin/env bash
# Official measurement: cold-start training for each seed, one at a time on this
# GPU, then offline evaluation of every run against the frozen bar.
#   scripts/run_official.sh <recipe> <budget_min> <out_dir> <seed> [seed...] [-- extra launch args]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
recipe="$1"; budget="$2"; out="$3"; shift 3
seeds=(); extra=()
while (( $# )); do
  if [[ "$1" == "--" ]]; then shift; extra=("$@"); break; fi
  seeds+=("$1"); shift
done
mkdir -p "$out"
for s in "${seeds[@]}"; do
  run="$out/${recipe}_s${s}"
  if [[ -f "$run/run.json" ]] && grep -q finished_ts "$run/run.json"; then
    echo "skip $run (done)"; continue
  fi
  rm -rf "$run"
  BS_COLD=1 "$here/scripts/train.sh" "$recipe" "$s" "$run" "$budget" "${extra[@]}" > "$run.log" 2>&1
done
for s in "${seeds[@]}"; do
  run="$out/${recipe}_s${s}"
  [[ -f "$run/time_to_bar.json" ]] || MUJOCO_GL=egl "$here/.venv/bin/python" -m babysteps.eval.time_to_bar --run "$run" --procs "${BS_EVAL_PROCS:-4}" > "$run.eval.log" 2>&1
  "$here/.venv/bin/python" -c "import json,sys; d=json.load(open('$run/time_to_bar.json')); print('$run', 'PASS' if d['passed'] else 'FAIL', round(d.get('time_to_bar_s', float('nan'))/60, 2), 'min')"
done
