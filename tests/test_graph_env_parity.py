"""Graphed env vs stock env, stepped in lockstep under a deterministic config.

Every source of randomness is pinned to a zero-width range (commands, reset pose,
push, startup randomization, observation noise), so the two implementations must
produce the same observations, rewards and dones for the same actions, including
across falls, resets and pushes.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.gpu


def _det_cfg(n_envs):
  import mjlab.tasks  # noqa: F401
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
  cfg.scene.num_envs = n_envs
  tw = cfg.commands["twist"]
  tw.heading_command = False
  tw.ranges.heading = None
  tw.rel_standing_envs = 0.0
  tw.rel_forward_envs = 0.0
  tw.rel_heading_envs = 0.0
  tw.ranges.lin_vel_x = (0.5, 0.5)
  tw.ranges.lin_vel_y = (-0.2, -0.2)
  tw.ranges.ang_vel_z = (0.3, 0.3)
  tw.resampling_time_range = (2.0, 2.0)
  cfg.events["reset_base"].params["pose_range"] = {"x": (0.1, 0.1), "y": (-0.1, -0.1), "z": (0.03, 0.03), "yaw": (0.5, 0.5)}
  cfg.events["push_robot"].params["velocity_range"] = {"x": (0.4, 0.4), "y": (-0.3, -0.3)}
  cfg.events["push_robot"].interval_range_s = (1.0, 1.0)
  cfg.events["foot_friction"].params["ranges"] = (0.8, 0.8)
  cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
  for ax in (0, 1, 2):
    cfg.events["base_com"].params["ranges"][ax] = (0.0, 0.0)
  cfg.observations["actor"].enable_corruption = False
  cfg.curriculum = {}
  return cfg


def test_graphed_step_matches_stock(steps: int = 400, n: int = 32, horizon: int = 30):
  """Stock mjlab is not bitwise deterministic (two identical stock envs diverge
  after ~46 steps, solver atomics + chaotic falls), so the checks are:
  1. before ``horizon`` steps: observations within 1e-3 and identical dones;
  2. at every step where an env resets in both envs: identical post-reset
     observation for that env (exercises the rewritten reset/command/event path).
  """
  from mjlab.envs import ManagerBasedRlEnv

  from babysteps.systems.graph_env import GraphedEnv, capture

  ref = ManagerBasedRlEnv(cfg=_det_cfg(n), device="cuda:0")
  ge = GraphedEnv(_det_cfg(n), n)
  act = torch.zeros(n, ge.num_actions, device="cuda:0")
  ge.reset_all()
  ge.step_body(act)  # eager warm-up: loads kernels, fills lazy caches
  ref.reset()
  ge.reset_all()
  g, out = capture(lambda: ge.step_body(act), warmup=0)
  gen = torch.Generator(device="cuda:0").manual_seed(1)
  early_obs, early_done_mismatch = 0.0, 0
  reset_pairs, reset_worst = 0, 0.0
  for t in range(steps):
    a = 1.2 * torch.randn(n, ge.num_actions, device="cuda:0", generator=gen)
    act.copy_(a)
    obs_r, _, term_r, tout_r, _ = ref.step(a)
    g.replay()
    obs_g, _, term_g, tout_g = out
    done_r, done_g = term_r | tout_r, (term_g | tout_g)[:n]
    if t < horizon:
      early_obs = max(early_obs, float((obs_r["actor"] - obs_g["actor"][:n]).abs().max()))
      early_done_mismatch += int((done_r != done_g).sum())
    both = done_r & done_g
    if bool(both.any()):
      d = (obs_r["actor"][both] - obs_g["actor"][:n][both]).abs().max()
      reset_worst = max(reset_worst, float(d))
      reset_pairs += int(both.sum())
  print({"early_obs": early_obs, "early_done_mismatch": early_done_mismatch,
         "reset_pairs": reset_pairs, "reset_worst": reset_worst})
  assert early_done_mismatch == 0
  assert early_obs < 1e-3, early_obs
  assert reset_pairs >= 20, "test must exercise the reset path"
  assert reset_worst < 1e-4, reset_worst


if __name__ == "__main__":
  test_graphed_step_matches_stock()
