"""Capture GraphedEnv.step_body alone on a small env and report the first failure."""
import sys
import traceback

import torch

from babysteps.systems.graph_env import GraphedEnv, capture


def main(n=64):
  import mjlab.tasks  # noqa: F401
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
  ge = GraphedEnv(cfg, n)
  ge.reset_all()
  print("patched ids:", ge.patched_ids, "raycast:", ge.patched_raycast)
  act = torch.zeros(n, ge.num_actions, device="cuda:0")
  try:
    g, out = capture(lambda: ge.step_body(act), warmup=2)
  except Exception:
    tb = traceback.format_exc().splitlines()
    print("\n".join([ln for ln in tb if "File" in ln or "Error" in ln][:30]))
    sys.exit(1)
  for i in range(200):
    act.copy_(0.5 * torch.randn_like(act))
    g.replay()
  torch.cuda.synchronize()
  obs, rew, term, tout = out
  print("CAPTURE OK", {k: tuple(v.shape) for k, v in obs.items()}, float(rew[:n].mean()),
        int(term[:n].sum()), "ep_len mean", float(ge.env.episode_length_buf[:n].float().mean()))


if __name__ == "__main__":
  main()
