"""Mirror map check against physics: obs(mirror(state)) == mirror(obs(state)).

Uses the CPU sim-to-sim bundle (needs bar/sim2sim/g1_flat.mjb; skipped if absent).
"""

import math
from pathlib import Path

import numpy as np
import pytest

from babysteps.eval.sim2sim import BUNDLE_DIR
from babysteps.systems.symmetry import _VEC_SIGNS, joint_mirror

pytestmark = pytest.mark.skipif(not (BUNDLE_DIR / "g1_flat.mjb").exists(), reason="model bundle not built")


def _mirror_maps(bundle):
  names = bundle.meta["joint_names"]
  jperm, jsign = joint_mirror(names)
  perm, sign, off = [], [], 0
  for term, d in bundle.obs_layout:
    if term in _VEC_SIGNS:
      perm += [off + i for i in range(d)]
      sign += list(_VEC_SIGNS[term])
    else:
      perm += [off + p for p in jperm]
      sign += jsign
    off += d
  return np.array(perm), np.array(sign), np.array(jperm), np.array(jsign)


def test_mirror_matches_physics():
  import mujoco

  from babysteps.eval.sim2sim import Bundle

  b = Bundle(Path(BUNDLE_DIR))
  m = b.m
  perm, sign, jperm, jsign = _mirror_maps(b)
  rng = np.random.default_rng(0)
  d = mujoco.MjData(m)
  b.reset(d, yaw=0.0)
  # random but physical state: joint offsets, base tilt/velocity, then let it move
  d.qpos[b.jq] += rng.uniform(-0.2, 0.2, len(b.jq))
  d.qvel[b.jv] = rng.uniform(-1.0, 1.0, len(b.jv))
  d.qvel[b.root_vadr : b.root_vadr + 6] = rng.uniform(-0.5, 0.5, 6)
  ax = np.array([0.3, 0.2, 0.1])
  ang = 0.2
  q = np.concatenate([[math.cos(ang / 2)], math.sin(ang / 2) * ax / np.linalg.norm(ax)])
  r = b.root_qadr
  d.qpos[r + 3 : r + 7] = q
  mujoco.mj_forward(m, d)
  last_a = rng.uniform(-1, 1, len(b.jq))
  cmd = np.array([0.4, 0.3, -0.2])
  obs = b.observation(d, last_a, cmd)

  dm = mujoco.MjData(m)
  dm.qpos[:] = d.qpos
  dm.qvel[:] = d.qvel
  dm.qpos[r + 1] = -d.qpos[r + 1]
  w, x, y, z = d.qpos[r + 3 : r + 7]
  dm.qpos[r + 3 : r + 7] = [w, -x, y, -z]
  v = b.root_vadr
  dm.qvel[v : v + 3] = d.qvel[v : v + 3] * [1, -1, 1]
  dm.qvel[v + 3 : v + 6] = d.qvel[v + 3 : v + 6] * [-1, 1, -1]
  q_rel = d.qpos[b.jq] - b.default_jpos
  dm.qpos[b.jq] = b.default_jpos + (q_rel[jperm] * jsign)
  dm.qvel[b.jv] = d.qvel[b.jv][jperm] * jsign
  mujoco.mj_forward(m, dm)
  obs_m = b.observation(dm, last_a[jperm] * jsign, cmd * [1, -1, -1])

  err = np.abs(obs[perm] * sign - obs_m)
  assert err.max() < 1e-6, (err.max(), int(err.argmax()))


if __name__ == "__main__":
  test_mirror_matches_physics()
  print("mirror OK")
