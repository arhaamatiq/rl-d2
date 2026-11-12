"""Throughput of the stock G1 flat env vs number of envs.

  python scripts/bench_env.py --envs 4096 8192 16384

For each size: physics-only step time (MuJoCo Warp graph replay, 4 substeps per
control step) and full env.step time (physics + mjlab managers), both synchronized.
"""

from __future__ import annotations

import argparse
import json
import time

import torch


def bench(n: int, steps: int = 60) -> dict:
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
  cfg.scene.num_envs = n
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  env.reset()
  na = env.action_manager.total_action_dim
  a = torch.zeros(n, na, device="cuda:0")
  for _ in range(10):
    env.step(a)
  torch.cuda.synchronize()
  t = time.perf_counter()
  for _ in range(steps):
    env.step(0.3 * torch.randn_like(a))
  torch.cuda.synchronize()
  full = (time.perf_counter() - t) / steps
  sim = env.sim
  torch.cuda.synchronize()
  t = time.perf_counter()
  for _ in range(steps):
    for _ in range(env.cfg.decimation):
      sim.step()
  torch.cuda.synchronize()
  phys = (time.perf_counter() - t) / steps
  mem = torch.cuda.max_memory_allocated() / 2**30
  env.close()
  del env
  torch.cuda.empty_cache()
  return {
    "num_envs": n,
    "env_step_ms": full * 1e3,
    "physics_ms": phys * 1e3,
    "managers_ms": (full - phys) * 1e3,
    "samples_per_s_env": n / full,
    "samples_per_s_physics_only": n / phys,
    "torch_max_mem_gb": mem,
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--envs", type=int, nargs="+", default=[4096, 8192, 16384])
  a = ap.parse_args()
  for n in a.envs:
    print(json.dumps(bench(n)), flush=True)


if __name__ == "__main__":
  main()
