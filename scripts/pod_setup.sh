#!/usr/bin/env bash
# Prepare a fresh Ubuntu 24.04 GPU machine (CUDA driver >= 12.8) for Babysteps.
#   scripts/pod_setup.sh [--nsys]
# Idempotent. Installs system libs, the locked Python env, and regenerates the
# frozen sim-to-sim model (verifying its pinned sha256).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

export DEBIAN_FRONTEND=noninteractive
if ! ldconfig -p | grep -q "libEGL.so "; then
  apt-get update -qq
  apt-get install -y -qq libegl-dev libgl1 libglvnd0 tmux >/dev/null
fi

if [[ "${1:-}" == "--nsys" ]] && ! command -v nsys >/dev/null; then
  if ! grep -rqs "developer.download.nvidia.com/compute/cuda/repos" /etc/apt/sources.list.d/; then
    apt-get install -y -qq wget gnupg >/dev/null
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/kr.deb
    dpkg -i /tmp/kr.deb >/dev/null
  fi
  apt-get update -qq || true
  pkg=$(apt-cache search --names-only '^nsight-systems-[0-9]' | awk '{print $1}' | sort -V | tail -1)
  apt-get install -y -qq "$pkg" >/dev/null
  ln -sf "$(ls -d /opt/nvidia/nsight-systems/*/bin | sort -V | tail -1)/nsys" /usr/local/bin/nsys
fi

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --locked --extra cu128
MUJOCO_GL=egl .venv/bin/python -m babysteps.eval.export_bundle --verify --device cpu
echo "setup ok"
