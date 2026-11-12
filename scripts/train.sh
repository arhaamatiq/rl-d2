#!/usr/bin/env bash
# Official training entry point.
#   scripts/train.sh <recipe> <seed> <run_dir> [budget_min] [extra launch args...]
#
# RLD2_COLD=1 (default) clears every compilation cache first (Warp kernels, Torch
# Inductor/Triton, CUDA JIT) so startup + compile are measured cold, the same
# way a fresh machine would see them. The launch clock starts after the cache
# wipe and before the interpreter starts.
set -euo pipefail

recipe="$1"; seed="$2"; run_dir="$3"; budget="${4:-90}"
shift $(( $# >= 4 ? 4 : $# ))

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="${RLD2_PYTHON:-$here/.venv/bin/python}"

# RLD2_CACHE_ROOT gives a run its own compilation caches, so two cold runs on one
# machine (one per GPU) never wipe or reuse each other's kernels.
if [[ -n "${RLD2_CACHE_ROOT:-}" ]]; then
  export WARP_CACHE_PATH="$RLD2_CACHE_ROOT/warp" TRITON_CACHE_DIR="$RLD2_CACHE_ROOT/triton" \
         TORCHINDUCTOR_CACHE_DIR="$RLD2_CACHE_ROOT/inductor" CUDA_CACHE_PATH="$RLD2_CACHE_ROOT/nv"
fi
if [[ "${RLD2_COLD:-1}" == "1" ]]; then
  rm -rf "${WARP_CACHE_PATH:-${HOME}/.cache/warp}" "${TRITON_CACHE_DIR:-${HOME}/.triton/cache}" \
         "${CUDA_CACHE_PATH:-${HOME}/.nv/ComputeCache}" "/tmp/torchinductor_$(id -un)" \
         "${TORCHINDUCTOR_CACHE_DIR:-/nonexistent}" 2>/dev/null || true
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export RLD2_COMMIT="${RLD2_COMMIT:-$(git -C "$here" rev-parse --short HEAD 2>/dev/null || echo unknown)}"
export RLD2_LAUNCH_TS="$(date +%s.%N)"
exec "$py" -m rl_d2.train.launch --recipe "$recipe" --seed "$seed" --out "$run_dir" \
  --budget-min "$budget" "$@"
