"""Export the frozen CPU sim-to-sim bundle from mjlab, and check observation parity.

Run once on a GPU machine:
  uv run python -m rl_d2.eval.export_bundle --out bar/sim2sim

The action -> ctrl map is measured rather than assumed: each action dimension is
perturbed through mjlab's own action pipeline and the resulting change in
``data.ctrl`` is recorded. That way the CPU replay cannot silently disagree with
how mjlab applies actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def make_nominal_env(device: str = "cuda:0"):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

  cfg = unitree_g1_flat_env_cfg(play=True)
  cfg.scene.num_envs = 1
  for k in ("foot_friction", "encoder_bias", "base_com", "push_robot"):
    cfg.events.pop(k, None)
  cfg.rewards = {}
  cfg.curriculum = {}
  return ManagerBasedRlEnv(cfg=cfg, device=device)


def _sensor_adr(m, suffix: str) -> int:
  import mujoco

  for i in range(m.nsensor):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) or ""
    if name.endswith(suffix):
      return int(m.sensor_adr[i])
  raise KeyError(suffix)


@torch.no_grad()
def measure_action_map(env) -> tuple[list[int], list[float], list[float]]:
  am = env.action_manager
  n = am.total_action_dim
  data = env.sim.data

  def ctrl_for(a: torch.Tensor) -> np.ndarray:
    am.process_action(a.view(1, -1))
    am.apply_action()
    env.scene.write_data_to_sim()
    return data.ctrl[0].detach().cpu().double().numpy().copy()

  base = ctrl_for(torch.zeros(n, device=env.device))
  idx, scale, offset = [], [], []
  for i in range(n):
    a = torch.zeros(n, device=env.device)
    a[i] = 1.0
    diff = ctrl_for(a) - base
    nz = np.flatnonzero(np.abs(diff) > 1e-9)
    assert len(nz) == 1, f"action {i} moved ctrl {nz}"
    c = int(nz[0])
    idx.append(c)
    scale.append(float(diff[c]))
    offset.append(float(base[c]))
  ctrl_for(torch.zeros(n, device=env.device))
  return idx, scale, offset


def export(out_dir: Path, device: str = "cuda:0") -> dict:
  import mujoco

  env = make_nominal_env(device)
  env.reset()
  m = env.sim.mj_model
  robot = env.scene["robot"]
  ix = robot.indexing
  out_dir.mkdir(parents=True, exist_ok=True)
  mujoco.mj_saveModel(m, str(out_dir / "g1_flat.mjb"))

  om = env.observation_manager
  names = om.active_terms["actor"]
  dims = [int(np.prod(d)) for d in om.group_obs_term_dim["actor"]]
  idx, scale, offset = measure_action_map(env)
  root_state = robot.data.default_root_state[0].detach().cpu().double().numpy()
  meta = {
    "mujoco_version": mujoco.__version__,
    "root_qadr": int(ix.free_joint_q_adr[0]),
    "root_vadr": int(ix.free_joint_v_adr[0]),
    "joint_names": list(robot.joint_names),
    "joint_qadr": [int(x) for x in ix.joint_q_adr.cpu().tolist()],
    "joint_vadr": [int(x) for x in ix.joint_v_adr.cpu().tolist()],
    "default_joint_pos": robot.data.default_joint_pos[0].cpu().double().tolist(),
    "action_ctrl_index": idx,
    "action_ctrl_scale": scale,
    "action_ctrl_offset": offset,
    "imu_lin_vel_adr": _sensor_adr(m, "imu_lin_vel"),
    "imu_ang_vel_adr": _sensor_adr(m, "imu_ang_vel"),
    "default_root_pos": root_state[0:3].tolist(),
    "default_root_quat": root_state[3:7].tolist(),
    "decimation": int(env.cfg.decimation),
    "timestep": float(m.opt.timestep),
    "obs_layout": [[n, d] for n, d in zip(names, dims, strict=True)],
  }
  from rl_d2.bar import file_sha256

  meta = {"mjb_sha256": file_sha256(out_dir / "g1_flat.mjb"), **meta}
  (out_dir / "g1_flat.json").write_text(json.dumps(meta, indent=2) + "\n")
  return meta


@torch.no_grad()
def parity_check(bundle_dir: Path, device: str = "cuda:0", steps: int = 50, seed: int = 0) -> dict:
  """Drive mjlab with random actions; at each step rebuild the CPU obs from the
  GPU state and compare with mjlab's actor observation."""
  import mujoco

  from rl_d2.eval.sim2sim import Bundle

  b = Bundle(bundle_dir)
  env = make_nominal_env(device)
  obs, _ = env.reset()
  robot = env.scene["robot"]
  d = mujoco.MjData(b.m)
  g = torch.Generator(device=device).manual_seed(seed)
  n = env.action_manager.total_action_dim
  last_a = np.zeros(n)
  max_obs_err, max_vel_err = 0.0, 0.0
  for _ in range(steps):
    d.qpos[:] = env.sim.data.qpos[0].cpu().double().numpy()
    d.qvel[:] = env.sim.data.qvel[0].cpu().double().numpy()
    d.ctrl[:] = env.sim.data.ctrl[0].cpu().double().numpy()
    mujoco.mj_forward(b.m, d)
    cmd = env.command_manager.get_command("twist")[0].cpu().double().numpy()
    cpu_obs = b.observation(d, last_a, cmd)
    gpu_obs = obs["actor"][0].cpu().double().numpy()
    max_obs_err = max(max_obs_err, float(np.abs(cpu_obs - gpu_obs).max()))
    lin_b, ang_b = b.root_vel_b(d)
    ref = torch.cat([robot.data.root_link_lin_vel_b[0], robot.data.root_link_ang_vel_b[0]])
    max_vel_err = max(
      max_vel_err, float(np.abs(np.concatenate([lin_b, ang_b]) - ref.cpu().double().numpy()).max())
    )
    a = 0.3 * torch.randn(1, n, device=device, generator=g)
    obs, *_ = env.step(a)
    last_a = a[0].cpu().double().numpy()
  return {"max_obs_abs_err": max_obs_err, "max_root_vel_abs_err": max_vel_err, "steps": steps}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=Path, default=Path("bar/sim2sim"))
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--parity-only", action="store_true")
  ap.add_argument("--verify", action="store_true", help="re-export to a temp dir and compare hashes")
  args = ap.parse_args()
  if args.verify:
    import tempfile

    pinned = json.loads((args.out / "g1_flat.json").read_text())["mjb_sha256"]
    with tempfile.TemporaryDirectory() as td:
      meta = export(Path(td), args.device)
      ok = meta["mjb_sha256"] == pinned
      if ok and not (args.out / "g1_flat.mjb").exists():
        import shutil

        shutil.copy(Path(td) / "g1_flat.mjb", args.out / "g1_flat.mjb")
    print(json.dumps({"pinned": pinned, "exported": meta["mjb_sha256"], "match": ok}))
    raise SystemExit(0 if ok else 1)
  if not args.parity_only:
    meta = export(args.out, args.device)
    print(json.dumps({k: v for k, v in meta.items() if not isinstance(v, list)}, indent=2))
  print(json.dumps(parity_check(args.out, args.device), indent=2))


if __name__ == "__main__":
  main()
