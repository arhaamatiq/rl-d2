"""GPU stages of the bar: velocity tracking, fall-free episodes and push recovery.

One batched rollout evaluates a checkpoint:
  * tracking envs: every bar command x ``reps_per_command``, 20 s, no disturbance
  * push envs:     structured commands x 8 directions x reps, one fixed push at 6 s

The environment is the stock mjlab flat G1 *play* configuration with a fixed
command table. Nothing in this file is allowed to depend on the training recipe.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from babysteps.bar import Bar
from babysteps.policy import PortablePolicy

# ----------------------------------------------------------------------------------
# Fixed command term (eval only)
# ----------------------------------------------------------------------------------


def _fixed_command_classes():
  from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
  )

  class FixedVelocityCommand(UniformVelocityCommand):
    """Velocity command that holds a per-env value from a fixed table."""

    def __init__(self, cfg, env):
      super().__init__(cfg, env)
      self._table = torch.tensor(cfg.table, dtype=torch.float32, device=self.device)
      assert self._table.shape == (self.num_envs, 3), self._table.shape

    def _resample_command(self, env_ids):
      self.vel_command_b[env_ids] = self._table[env_ids]
      self.vel_command_w[env_ids] = self._table[env_ids]
      for flag in (
        self.is_heading_env,
        self.is_standing_env,
        self.is_world_env,
        self.is_forward_env,
      ):
        flag[env_ids] = False

    def _update_command(self, env_ids=None):
      del env_ids

  @dataclass(kw_only=True)
  class FixedVelocityCommandCfg(UniformVelocityCommandCfg):
    table: tuple = ()

    def build(self, env):
      return FixedVelocityCommand(self, env)

  return FixedVelocityCommand, FixedVelocityCommandCfg


# ----------------------------------------------------------------------------------
# Layout of the batched evaluation
# ----------------------------------------------------------------------------------


@dataclass
class Layout:
  commands: np.ndarray  # (N, 3)
  is_push: np.ndarray  # (N,) bool
  push_dir: np.ndarray  # (N,) radians, relative to heading
  cmd_index: np.ndarray  # (N,) index into the tracking command list (-1 for push envs)
  track_commands: list = field(default_factory=list)

  @property
  def num_envs(self) -> int:
    return int(self.commands.shape[0])


def build_layout(bar: Bar) -> Layout:
  tr = bar.tracking
  track_cmds = list(tr["commands_structured"]) + list(tr["commands_random"])
  reps = int(tr["reps_per_command"])
  cmds, is_push, pdir, cidx = [], [], [], []
  for i, c in enumerate(track_cmds):
    for _ in range(reps):
      cmds.append(c)
      is_push.append(False)
      pdir.append(0.0)
      cidx.append(i)
  pu = bar.push
  push_cmds = list(tr["commands_structured"]) if pu["commands"] == "structured" else pu["commands"]
  ndir = int(pu["directions"])
  for c in push_cmds:
    for k in range(ndir):
      for _ in range(int(pu["reps"])):
        cmds.append(c)
        is_push.append(True)
        pdir.append(2.0 * math.pi * k / ndir)
        cidx.append(-1)
  return Layout(
    commands=np.asarray(cmds, dtype=np.float32),
    is_push=np.asarray(is_push, dtype=bool),
    push_dir=np.asarray(pdir, dtype=np.float32),
    cmd_index=np.asarray(cidx, dtype=np.int64),
    track_commands=track_cmds,
  )


# ----------------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------------


def make_eval_env_cfg(bar: Bar, layout: Layout):
  """Stock flat G1 play config + fixed commands. Frozen together with the bar."""
  from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
  from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

  _, FixedVelocityCommandCfg = _fixed_command_classes()

  cfg = unitree_g1_flat_env_cfg(play=True)
  cfg.scene.num_envs = layout.num_envs
  cfg.seed = int(bar.tracking["seed"])
  # Long enough that no env times out inside the 20 s window.
  cfg.episode_length_s = float(bar.tracking["episode_s"]) + 5.0
  cfg.commands = {
    "twist": FixedVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      heading_command=False,
      rel_standing_envs=0.0,
      rel_heading_envs=0.0,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-3.0, 3.0), lin_vel_y=(-3.0, 3.0), ang_vel_z=(-3.0, 3.0), heading=None
      ),
      debug_vis=False,
      table=tuple(tuple(float(v) for v in c) for c in layout.commands),
    )
  }
  # Falls are detected by the evaluator; the env must not auto-reset a fallen robot.
  cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "time_out"}
  cfg.rewards = {}
  cfg.curriculum = {}
  cfg.events.pop("push_robot", None)
  return cfg


@dataclass
class EvalEnv:
  env: object
  layout: Layout
  bar: Bar
  device: str

  @staticmethod
  def create(bar: Bar, device: str = "cuda:0") -> "EvalEnv":
    from mjlab.envs import ManagerBasedRlEnv

    layout = build_layout(bar)
    cfg = make_eval_env_cfg(bar, layout)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    return EvalEnv(env=env, layout=layout, bar=bar, device=device)


# ----------------------------------------------------------------------------------
# Rollout
# ----------------------------------------------------------------------------------


@torch.no_grad()
def run_gpu_eval(ev: EvalEnv, policy: PortablePolicy, record_video_env: int | None = None) -> dict:
  bar, lay, env, dev = ev.bar, ev.layout, ev.env, ev.device
  t0 = time.time()
  ctrl_dt = env.step_dt
  n_steps = int(round(bar.tracking["episode_s"] / ctrl_dt))
  settle = int(round(bar.tracking["settle_s"] / ctrl_dt))
  push_step = int(round(bar.push["push_time_s"] / ctrl_dt))
  dv = bar.push.get("delta_v_mps")
  tilt_lim = math.radians(bar.fall["tilt_deg"])
  min_h = float(bar.fall["min_pelvis_height_m"])

  # Deterministic reset: identical initial states for every checkpoint.
  env.seed(int(bar.tracking["seed"]))
  obs, _ = env.reset()
  pol = policy.to_torch(dev)
  robot = env.scene["robot"]

  N = lay.num_envs
  cmd = torch.as_tensor(lay.commands, device=dev)
  is_push = torch.as_tensor(lay.is_push, device=dev)
  push_ids = torch.nonzero(is_push).flatten()
  push_dir = torch.as_tensor(lay.push_dir, device=dev)

  alive = torch.ones(N, dtype=torch.bool, device=dev)
  fall_step = torch.full((N,), n_steps, dtype=torch.long, device=dev)
  lin_sum = torch.zeros(N, device=dev)
  yaw_sum = torch.zeros(N, device=dev)
  cnt = torch.zeros(N, device=dev)
  alive_at_push = None

  frames = [] if record_video_env is not None else None
  for t in range(n_steps):
    if t == push_step and dv is not None and len(push_ids) > 0:
      vel_w = robot.data.root_link_vel_w[push_ids].clone()
      ang = robot.data.heading_w[push_ids] + push_dir[push_ids]
      vel_w[:, 0] += dv * torch.cos(ang)
      vel_w[:, 1] += dv * torch.sin(ang)
      robot.write_root_link_velocity_to_sim(vel_w, env_ids=push_ids)
      alive_at_push = alive.clone()

    act = pol(obs["actor"])
    obs, _, _, _, _ = env.step(act)

    g = robot.data.projected_gravity_b
    tilt = torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0))
    z = robot.data.root_link_pos_w[:, 2]
    fell = alive & ((tilt > tilt_lim) | (z < min_h))
    fall_step = torch.where(fell, torch.full_like(fall_step, t), fall_step)
    alive = alive & ~fell

    if t >= settle:
      v = robot.data.root_link_lin_vel_b
      w = robot.data.root_link_ang_vel_b
      le = torch.linalg.norm(cmd[:, :2] - v[:, :2], dim=-1)
      ye = torch.abs(cmd[:, 2] - w[:, 2])
      m = alive.float()
      lin_sum += le * m
      yaw_sum += ye * m
      cnt += m

    if frames is not None:
      frames.append(env.render())
    if t % 50 == 0 and not bool(alive.any()):
      break  # every robot is down; no need to finish the episode

  track = ~is_push
  ep_lin = (lin_sum / cnt.clamp(min=1.0)).cpu().numpy()
  ep_yaw = (yaw_sum / cnt.clamp(min=1.0)).cpu().numpy()
  ep_fall_free = alive.cpu().numpy()
  tr = track.cpu().numpy()
  # Episodes that fell before collecting any post-settle sample carry no tracking
  # signal; they count as a tracking error equal to the full command norm.
  cmd_np = lay.commands
  no_sample = (cnt.cpu().numpy() == 0)
  ep_lin = np.where(no_sample, np.linalg.norm(cmd_np[:, :2], axis=1), ep_lin)
  ep_yaw = np.where(no_sample, np.abs(cmd_np[:, 2]), ep_yaw)

  res = {
    "lin_err": float(ep_lin[tr].mean()),
    "yaw_err": float(ep_yaw[tr].mean()),
    "fall_free": float(ep_fall_free[tr].mean()),
    "n_track": int(tr.sum()),
  }
  if dv is not None and len(push_ids) > 0:
    pm = lay.is_push
    res["push_success"] = float(ep_fall_free[pm].mean())
    res["push_alive_before"] = float(alive_at_push.cpu().numpy()[pm].mean()) if alive_at_push is not None else 0.0
    res["n_push"] = int(pm.sum())
  per_cmd = []
  for i, c in enumerate(lay.track_commands):
    sel = lay.cmd_index == i
    per_cmd.append({
      "cmd": [float(x) for x in c],
      "lin_err": float(ep_lin[sel].mean()),
      "yaw_err": float(ep_yaw[sel].mean()),
      "fall_free": float(ep_fall_free[sel].mean()),
    })
  res["per_command"] = per_cmd
  res["gpu_eval_s"] = time.time() - t0
  if frames is not None:
    res["frames"] = frames
  return res
