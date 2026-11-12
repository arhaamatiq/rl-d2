"""Side-by-side video at equal wall-clock training time.

  MUJOCO_GL=egl python scripts/make_video.py \
      --left runs/official/stock_s0 --right runs/official/track_a_s0 --minute 5 \
      --out videos/minute5.mp4

For each run, picks the last checkpoint written at or before ``--minute`` minutes of
wall-clock training (launch clock), rolls it out in CPU MuJoCo (the sim-to-sim
model) with a fixed command schedule and one push, renders a tracking camera, and
stacks the two clips horizontally with labels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from babysteps import policy as policy_io
from babysteps.eval.sim2sim import Bundle, quat_rotate_inverse, yaw_from_quat

SCHEDULE = [(0.0, (0.0, 0.0, 0.0)), (1.5, (0.6, 0.0, 0.0)), (5.0, (1.0, 0.0, 0.0)),
            (8.0, (0.5, 0.0, 0.4)), (11.0, (0.0, 0.4, 0.0)), (13.0, (0.8, 0.0, 0.0))]
PUSH_T, PUSH_DV = 9.5, 0.8


def pick_checkpoint(run: Path, minute: float) -> tuple[Path, dict]:
  rows = [json.loads(x) for x in (run / "checkpoints.jsonl").read_text().splitlines() if x]
  ok = [r for r in rows if r["t_wall_s"] <= minute * 60.0]
  r = max(ok, key=lambda r: r["t_wall_s"]) if ok else rows[0]
  return run / r["path"], r


def rollout_frames(bundle: Bundle, pol, seconds: float, w: int, h: int) -> tuple[list[np.ndarray], int | None]:
  import mujoco

  m = bundle.m
  d = mujoco.MjData(m)
  bundle.reset(d, yaw=0.0)
  ren = mujoco.Renderer(m, height=h, width=w)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  pelvis = [i for i in range(m.nbody) if m.body(i).name.endswith("pelvis")]
  cam.trackbodyid = pelvis[0] if pelvis else 1
  cam.distance, cam.elevation, cam.azimuth = 3.2, -12.0, 120.0
  ctrl_dt = m.opt.timestep * bundle.decimation
  last_a = np.zeros(pol.act_dim)
  frames, fell = [], None
  r = bundle.root_qadr
  for t in range(int(seconds / ctrl_dt)):
    now = t * ctrl_dt
    cmd = np.array([c for (ts, c) in SCHEDULE if ts <= now][-1])
    if abs(now - PUSH_T) < ctrl_dt / 2:
      yaw = yaw_from_quat(d.qpos[r + 3 : r + 7])
      d.qvel[bundle.root_vadr] += PUSH_DV * math.cos(yaw + math.pi / 2)
      d.qvel[bundle.root_vadr + 1] += PUSH_DV * math.sin(yaw + math.pi / 2)
    obs = bundle.observation(d, last_a, cmd)
    a = pol.act_np(obs)
    for _ in range(bundle.decimation):
      bundle.apply_action(d, a)
      mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    last_a = a
    g = quat_rotate_inverse(d.qpos[r + 3 : r + 7], np.array([0.0, 0.0, -1.0]))
    if fell is None and (math.acos(max(-1.0, min(1.0, -g[2]))) > math.radians(70) or d.qpos[r + 2] < 0.5):
      fell = t
    if t % 2 == 0:  # 25 fps
      ren.update_scene(d, camera=cam)
      frames.append(ren.render().copy())
  ren.close()
  return frames, fell


def label(frames, text: str, sub: str):
  from PIL import Image, ImageDraw

  out = []
  for f in frames:
    im = Image.fromarray(f)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, im.width, 44], fill=(0, 0, 0))
    dr.text((10, 6), text, fill=(255, 255, 255))
    dr.text((10, 24), sub, fill=(200, 200, 200))
    out.append(np.asarray(im))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--left", type=Path, required=True)
  ap.add_argument("--right", type=Path, required=True)
  ap.add_argument("--left-name", default="stock PPO")
  ap.add_argument("--right-name", default="Babysteps (Track A)")
  ap.add_argument("--minute", type=float, default=5.0)
  ap.add_argument("--seconds", type=float, default=16.0)
  ap.add_argument("--out", type=Path, required=True)
  a = ap.parse_args()
  import mediapy

  bundle = Bundle()
  clips = []
  for run, name in ((a.left, a.left_name), (a.right, a.right_name)):
    path, row = pick_checkpoint(run, a.minute)
    pol = policy_io.load(path)
    frames, fell = rollout_frames(bundle, pol, a.seconds, 480, 400)
    status = f"fell at {fell * 0.02:.1f} s" if fell is not None else "no fall"
    sub = f"checkpoint at {row['t_wall_s']/60:.1f} min wall-clock, iteration {row['iteration']} | {status}"
    clips.append(label(frames, f"{name} - after {a.minute:g} min of training", sub))
    print(name, path, status, flush=True)
  n = min(len(c) for c in clips)
  video = [np.concatenate([clips[0][i], clips[1][i]], axis=1) for i in range(n)]
  a.out.parent.mkdir(parents=True, exist_ok=True)
  mediapy.write_video(str(a.out), video, fps=25)
  print("wrote", a.out)


if __name__ == "__main__":
  main()
