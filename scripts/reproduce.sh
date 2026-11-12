#!/usr/bin/env bash
# One-command reproduction (also the Docker entrypoint).
#   scripts/reproduce.sh headline   # Track A record: 5 seeds, cold start, eval, median/IQR
#   scripts/reproduce.sh baseline   # stock T0: 5 seeds
#   scripts/reproduce.sh quick      # 1 seed of the headline recipe (~smoke test)
# Results land in runs/repro/ and a summary in runs/repro/summary.json.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
HEADLINE_RECIPE="${HEADLINE_RECIPE:-track_a}"
HEADLINE_BUDGET="${HEADLINE_BUDGET:-30}"
BASELINE_BUDGET="${BASELINE_BUDGET:-75}"
SEEDS=(0 1 2 3 4)
out=runs/repro
mkdir -p "$out"

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
[[ -f bar/sim2sim/g1_flat.mjb ]] || MUJOCO_GL=egl .venv/bin/python -m babysteps.eval.export_bundle --verify --device cpu

case "${1:-headline}" in
  headline) scripts/run_official.sh "$HEADLINE_RECIPE" "$HEADLINE_BUDGET" "$out" "${SEEDS[@]}" ;;
  baseline) scripts/run_official.sh stock "$BASELINE_BUDGET" "$out" "${SEEDS[@]}" ;;
  quick)    scripts/run_official.sh "$HEADLINE_RECIPE" "$HEADLINE_BUDGET" "$out" 0 ;;
  *) echo "usage: $0 [headline|baseline|quick]"; exit 2 ;;
esac

.venv/bin/python scripts/summarize.py "$out"/*/ --json "$out/summary.json"
