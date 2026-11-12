"""Sim-to-sim stage of the bar: replay the policy in CPU MuJoCo.

Training and the GPU stages use MuJoCo Warp. Here the same policy drives the same
compiled model in the C MuJoCo engine, single world, no domain randomization.
Solver, contact pipeline and floating point paths differ, so a policy that only
works because of simulator quirks shows up here.

The bundle (``bar/sim2sim/``) is exported once from mjlab and frozen:
  g1_flat.mjb   compiled MjModel (binary, exact)
  g1_flat.json  index maps, default pose, action scale, observation layout
The parity test (tests/test_sim2sim_parity.py) checks that the CPU observation
matches mjlab's observation for the same state.
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from rl_d2.bar import BAR_DIR, Bar, file_sha256
from rl_d2.policy import PortablePolicy

BUNDLE_DIR = BAR_DIR / "sim2sim"


# ----------------------------------------------------------------------------------
# Quaternion helpers (MuJoCo convention: w, x, y, z)
# ----------------------------------------------------------------------------------


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  w, x, y, z = q
  r = np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ])
  return r.T @ v


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return np.array([
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ])


def yaw_from_quat(q: np.ndarray) -> float:
  w, x, y, z = q
  return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ----------------------------------------------------------------------------------
# Bundle
# ----------------------------------------------------------------------------------


class Bundle:
  def __init__(self, bundle_dir: Path = BUNDLE_DIR):
    import mujoco

    self.dir = Path(bundle_dir)
    self.meta = json.loads((self.dir / "g1_flat.json").read_text())
    self.model_path = self.dir / "g1_flat.mjb"
    if not self.model_path.exists():
      raise FileNotFoundError(
        f"{self.model_path} missing; run `python -m rl_d2.eval.export_bundle --verify`"
      )
    self.model_sha = file_sha256(self.model_path)
    if self.model_sha != self.meta["mjb_sha256"]:
      raise RuntimeError(f"sim2sim model hash mismatch: {self.model_sha} != {self.meta['mjb_sha256']}")
    self.m = mujoco.MjModel.from_binary_path(str(self.model_path))
    md = self.meta
    self.root_qadr = int(md["root_qadr"])
    self.root_vadr = int(md["root_vadr"])
    self.jq = np.asarray(md["joint_qadr"], dtype=np.int64)
    self.jv = np.asarray(md["joint_vadr"], dtype=np.int64)
    self.default_jpos = np.asarray(md["default_joint_pos"], dtype=np.float64)
    self.ctrl_idx = np.asarray(md["action_ctrl_index"], dtype=np.int64)
    self.ctrl_scale = np.asarray(md["action_ctrl_scale"], dtype=np.float64)
    self.ctrl_offset = np.asarray(md["action_ctrl_offset"], dtype=np.float64)
    self.imu_lin_adr = int(md["imu_lin_vel_adr"])
    self.imu_ang_adr = int(md["imu_ang_vel_adr"])
    self.default_root_pos = np.asarray(md["default_root_pos"], dtype=np.float64)
    self.default_root_quat = np.asarray(md["default_root_quat"], dtype=np.float64)
    self.decimation = int(md["decimation"])
    self.obs_layout = md["obs_layout"]  # [[name, dim], ...]

  def reset(self, d, yaw: float, z_offset: float = 0.03) -> None:
    import mujoco

    mujoco.mj_resetData(self.m, d)
    r = self.root_qadr
    d.qpos[r : r + 3] = self.default_root_pos + np.array([0.0, 0.0, z_offset])
    qyaw = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    d.qpos[r + 3 : r + 7] = quat_mul(qyaw, self.default_root_quat)
    d.qpos[self.jq] = self.default_jpos
    d.qvel[:] = 0.0
    mujoco.mj_forward(self.m, d)

  def observation(self, d, last_action: np.ndarray, cmd: np.ndarray) -> np.ndarray:
    r = self.root_qadr
    quat = d.qpos[r + 3 : r + 7]
    parts = {
      "base_lin_vel": d.sensordata[self.imu_lin_adr : self.imu_lin_adr + 3],
      "base_ang_vel": d.sensordata[self.imu_ang_adr : self.imu_ang_adr + 3],
      "projected_gravity": quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0])),
      "joint_pos": d.qpos[self.jq] - self.default_jpos,
      "joint_vel": d.qvel[self.jv],
      "actions": last_action,
      "command": cmd,
    }
    out = []
    for name, dim in self.obs_layout:
      v = np.asarray(parts[name], dtype=np.float64)
      assert v.shape == (dim,), (name, v.shape, dim)
      out.append(v)
    return np.concatenate(out)

  def root_vel_b(self, d) -> tuple[np.ndarray, np.ndarray]:
    """Root link linear / angular velocity in the body frame."""
    r, v = self.root_qadr, self.root_vadr
    quat = d.qpos[r + 3 : r + 7]
    lin_b = quat_rotate_inverse(quat, d.qvel[v : v + 3])
    ang_b = d.qvel[v + 3 : v + 6].copy()  # free-joint angular velocity is body-frame
    return lin_b, ang_b

  def apply_action(self, d, action: np.ndarray) -> None:
    d.ctrl[self.ctrl_idx] = self.ctrl_offset + self.ctrl_scale * action


# ----------------------------------------------------------------------------------
# Episodes
# ----------------------------------------------------------------------------------


def run_episode(
  bundle: Bundle,
  policy: PortablePolicy,
  cmd,
  yaw: float,
  episode_s: float,
  settle_s: float,
  tilt_deg: float,
  min_h: float,
) -> dict:
  import mujoco

  m = bundle.m
  d = mujoco.MjData(m)
  bundle.reset(d, yaw)
  cmd = np.asarray(cmd, dtype=np.float64)
  ctrl_dt = m.opt.timestep * bundle.decimation
  n_steps = int(round(episode_s / ctrl_dt))
  settle = int(round(settle_s / ctrl_dt))
  tilt_lim = math.radians(tilt_deg)
  last_a = np.zeros(policy.act_dim)
  lin_errs, yaw_errs = [], []
  fell_at = None
  for t in range(n_steps):
    obs = bundle.observation(d, last_a, cmd)
    a = policy.act_np(obs)
    for _ in range(bundle.decimation):
      bundle.apply_action(d, a)
      mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    last_a = a
    r = bundle.root_qadr
    g = quat_rotate_inverse(d.qpos[r + 3 : r + 7], np.array([0.0, 0.0, -1.0]))
    tilt = math.acos(max(-1.0, min(1.0, -g[2])))
    if tilt > tilt_lim or d.qpos[r + 2] < min_h or not np.isfinite(d.qpos).all():
      fell_at = t
      break
    if t >= settle:
      lin_b, ang_b = bundle.root_vel_b(d)
      lin_errs.append(float(np.linalg.norm(cmd[:2] - lin_b[:2])))
      yaw_errs.append(float(abs(cmd[2] - ang_b[2])))
  return {
    "cmd": cmd.tolist(),
    "yaw0": yaw,
    "fell_at_step": fell_at,
    "lin_err": float(np.mean(lin_errs)) if lin_errs else float(np.linalg.norm(cmd[:2])),
    "yaw_err": float(np.mean(yaw_errs)) if yaw_errs else float(abs(cmd[2])),
  }


_BUNDLE_CACHE: dict[str, Bundle] = {}


def _get_bundle(bundle_dir: str) -> Bundle:
  if bundle_dir not in _BUNDLE_CACHE:
    _BUNDLE_CACHE[bundle_dir] = Bundle(Path(bundle_dir))
  return _BUNDLE_CACHE[bundle_dir]


def _episode_worker(args):
  bundle_dir, policy_dict, cmd, yaw, episode_s, settle_s, tilt, min_h = args
  bundle = _get_bundle(bundle_dir)
  policy = PortablePolicy.from_dict(policy_dict)
  return run_episode(bundle, policy, cmd, yaw, episode_s, settle_s, tilt, min_h)


_POOL = None


def run_sim2sim(bar: Bar, policy: PortablePolicy, workers: int = 8, bundle_dir: Path = BUNDLE_DIR) -> dict:
  t0 = time.time()
  s = bar.sim2sim
  cmds = list(bar.tracking["commands_structured"]) if s["commands"] == "structured" else s["commands"]
  jobs = [
    (
      str(bundle_dir),
      policy.to_dict(),
      c,
      float(y),
      float(s["episode_s"]),
      float(bar.tracking["settle_s"]),
      float(bar.fall["tilt_deg"]),
      float(bar.fall["min_pelvis_height_m"]),
    )
    for c in cmds
    for y in s["initial_yaws_rad"]
  ]
  if workers > 1:
    # spawn, not fork: the parent process already holds a CUDA context.
    global _POOL
    if _POOL is None:
      import multiprocessing as mp

      _POOL = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    eps = list(_POOL.map(_episode_worker, jobs))
  else:
    eps = [_episode_worker(j) for j in jobs]
  thr = bar.thresholds.get("sim2sim_lin_err_max")
  ok = [
    (e["fell_at_step"] is None) and (thr is None or e["lin_err"] <= thr) for e in eps
  ]
  return {
    "sim2sim_success": float(np.mean(ok)),
    "sim2sim_fall_free": float(np.mean([e["fell_at_step"] is None for e in eps])),
    "sim2sim_lin_err": float(np.mean([e["lin_err"] for e in eps])),
    "sim2sim_yaw_err": float(np.mean([e["yaw_err"] for e in eps])),
    "sim2sim_episodes": eps,
    "sim2sim_s": time.time() - t0,
  }
