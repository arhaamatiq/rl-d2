#!/usr/bin/env bash
# Before/after Nsight Systems profiles on the current host, same procedure for both.
#   scripts/profile_pair.sh <track_a_recipe> [extra --set args for track A...]
# Writes profiles/{stock,track_a}.nsys-rep, .sqlite and results/profiles/*.json.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
ta="$1"; shift || true
mkdir -p profiles results/profiles runs/prof
export BS_COLD=0 BS_NVTX=1 MUJOCO_GL=egl

prof() {  # name recipe iters extra...
  local name="$1" recipe="$2" iters="$3"; shift 3
  rm -rf "runs/prof/$name"
  nsys profile -t cuda,nvtx,osrt -s none --cuda-memory-usage=false \
    --capture-range=cudaProfilerApi --capture-range-end=stop -f true -o "profiles/$name" \
    scripts/train.sh "$recipe" 0 "runs/prof/$name" 15 --profile-iters "$iters" "$@" > "runs/prof/$name.log" 2>&1
  nsys export --type sqlite -f true -o "profiles/$name.sqlite" "profiles/$name.nsys-rep" > /dev/null
  .venv/bin/python scripts/profile_report.py "profiles/$name.sqlite" --json "results/profiles/$name.json" > /dev/null
  echo "$name: $(python3 -c "import json; d=json.load(open('results/profiles/$name.json')); print(round(d['s_per_iter'],3), 's/iter, kernel coverage', round(d['kernel_coverage'],3), ', kernels/iter', round(d['kernels_per_iter']), ', syncs/iter', round(d['sync_api_calls_per_iter']))")"
}

prof stock stock 100:110
prof track_a "$ta" 50:60 "$@"
BS_NVTX=0 .venv/bin/python scripts/bench_components.py --recipe "$ta" "$@" --json results/profiles/components_track_a.json > /dev/null
BS_NVTX=0 .venv/bin/python scripts/bench_components.py --recipe graph_stock --json results/profiles/components_graph_stock.json > /dev/null
echo done
