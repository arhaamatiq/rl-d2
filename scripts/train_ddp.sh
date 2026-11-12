#!/usr/bin/env bash
# Data-parallel version of scripts/train.sh (same cold-cache and launch-clock rules).
#   scripts/train_ddp.sh <ngpus> <recipe> <seed> <run_dir> [budget_min] [extra launch args...]
# Rank 0 writes <run_dir>; rank r > 0 writes <run_dir>_rank<r>. Each rank simulates the
# recipe's num_envs; gradients and the adaptive-LR KL are all-reduced every minibatch.
set -euo pipefail
n="$1"; recipe="$2"; seed="$3"; run_dir="$4"; budget="${5:-90}"
shift $(( $# >= 5 ? 5 : $# ))
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${BS_COLD:-1}" == "1" ]]; then
  rm -rf "${HOME}/.cache/warp" "${HOME}/.triton/cache" "${HOME}/.nv/ComputeCache" \
         "/tmp/torchinductor_$(id -un)" 2>/dev/null || true
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export BS_COMMIT="${BS_COMMIT:-$(git -C "$here" rev-parse --short HEAD 2>/dev/null || echo unknown)}"
export BS_LAUNCH_TS="$(date +%s.%N)"
exec "$here/.venv/bin/torchrun" --standalone --nproc_per_node "$n" -m babysteps.train.launch \
  --recipe "$recipe" --seed "$seed" --out "$run_dir" --budget-min "$budget" "$@"
