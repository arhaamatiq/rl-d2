"""Bisect the rollout-graph crash: capture K env steps (no PPO) at N envs."""
import sys

import torch

from babysteps.systems.graph_env import GraphedEnv, capture


def main(n: int, k: int):
  import mjlab.tasks  # noqa: F401
  from mjlab.tasks.registry import load_env_cfg

  ge = GraphedEnv(load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1"), n)
  ge.reset_all()
  act = torch.zeros(n, ge.num_actions, device="cuda:0")

  def body():
    for _ in range(k):
      ge.step_body(act)

  g, _ = capture(body, warmup=1)
  print(f"captured n={n} k={k}; mem {torch.cuda.memory_reserved()/2**30:.2f} GiB", flush=True)
  for _ in range(5):
    g.replay()
  torch.cuda.synchronize()
  print(f"REPLAY OK n={n} k={k}", flush=True)


if __name__ == "__main__":
  main(int(sys.argv[1]), int(sys.argv[2]))
