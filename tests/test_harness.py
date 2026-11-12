import json
import time

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from babysteps import policy as policy_io
from babysteps.train.harness import CheckpointWriter, RunClock


def _actor() -> MLPModel:
  obs = TensorDict({"actor": torch.randn(4, 7)}, batch_size=[4])
  return MLPModel(
    obs,
    {"actor": ["actor"]},
    "actor",
    3,
    hidden_dims=(16, 16),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
  )


def test_due_uses_fake_clock(tmp_path, monkeypatch):
  now = [1000.0]
  monkeypatch.setattr(time, "time", lambda: now[0])
  clock = RunClock(t_launch=1000.0)
  w = CheckpointWriter(tmp_path, clock, interval_s=30.0)
  assert not w.due()
  now[0] = 1029.9
  assert not w.due()
  now[0] = 1030.0
  assert w.due()
  w.write(_actor(), iteration=1, env_steps=100)
  assert not w.due()  # reference moves to the last checkpoint
  now[0] = 1059.0
  assert not w.due()
  now[0] = 1060.0
  assert w.due()


def test_write_emits_rows_and_loadable_policy(tmp_path):
  clock = RunClock(t_launch=time.time())
  w = CheckpointWriter(tmp_path, clock, interval_s=1.0)
  actor = _actor()
  for i in range(3):
    row = w.write(actor, iteration=i, env_steps=10 * i, extra={"tag": "x"})
    assert row["k"] == i
  rows = [json.loads(line) for line in (tmp_path / "checkpoints.jsonl").read_text().splitlines()]
  assert [r["k"] for r in rows] == [0, 1, 2]
  assert all(r["tag"] == "x" for r in rows)
  p = policy_io.load(tmp_path / rows[-1]["path"])
  assert p.obs_dim == 7 and p.act_dim == 3
