import re

import numpy as np

from babysteps.bar import load_bar
from babysteps.eval import gpu_eval


def test_bar_commands():
  bar = load_bar()
  tr = bar.tracking
  assert len(tr["commands_structured"]) == 8
  assert len(tr["commands_random"]) == 16
  for c in tr["commands_structured"] + tr["commands_random"]:
    assert len(c) == 3
  for vx, vy, wz in tr["commands_random"]:
    assert -1.0 <= vx <= 1.0
    assert -1.0 <= vy <= 1.0
    assert -0.5 <= wz <= 0.5
  assert re.fullmatch(r"[0-9a-f]{64}", bar.sha256)


def test_layout_counts():
  bar = load_bar()
  layout = gpu_eval.build_layout(bar)
  reps = int(bar.tracking["reps_per_command"])
  n_track = 24 * reps
  n_push = 8 * int(bar.push["directions"]) * int(bar.push["reps"])
  assert layout.num_envs == n_track + n_push
  assert int((~layout.is_push).sum()) == n_track
  assert int(layout.is_push.sum()) == n_push
  assert layout.commands.shape == (layout.num_envs, 3)
  assert np.all(layout.cmd_index[~layout.is_push] >= 0)
  assert np.all(layout.cmd_index[layout.is_push] == -1)
