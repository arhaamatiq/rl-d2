"""Run harness shared by every track: launch clock, wall-clock checkpoints, budget.

Time-to-bar is measured from the *launch timestamp* (written by the shell wrapper
before the interpreter starts) to the first checkpoint that passes the bar.
Checkpoints are taken on a fixed wall-clock cadence (bar ``checkpoint_interval_s``)
so every track has the same measurement granularity, whatever its iteration speed.

A checkpoint is a PortablePolicy (weights + observation normalizer) plus a row in
``checkpoints.jsonl`` with its timestamps. Writing one costs a device->host copy of
~0.6 MB; that cost is paid identically by every track.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from babysteps import policy as policy_io


def process_start_time() -> float:
  """Unix time at which this process was created (Linux), else now."""
  try:
    with open("/proc/self/stat") as f:
      fields = f.read().rsplit(")", 1)[1].split()
    start_ticks = int(fields[19])
    with open("/proc/uptime") as f:
      uptime = float(f.read().split()[0])
    hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    return time.time() - uptime + start_ticks / hz
  except Exception:
    return time.time()


def launch_timestamp() -> float:
  ts = os.environ.get("BS_LAUNCH_TS")
  return float(ts) if ts else process_start_time()


@dataclass
class RunClock:
  t_launch: float
  t_first_iter: float | None = None
  t_last_ckpt: float | None = None
  events: dict = field(default_factory=dict)

  def mark(self, name: str) -> None:
    self.events[name] = time.time() - self.t_launch

  def wall(self) -> float:
    return time.time() - self.t_launch


class CheckpointWriter:
  def __init__(self, run_dir: Path, clock: RunClock, interval_s: float, activation: str = "elu"):
    self.run_dir = Path(run_dir)
    self.ckpt_dir = self.run_dir / "checkpoints"
    self.ckpt_dir.mkdir(parents=True, exist_ok=True)
    self.index_path = self.run_dir / "checkpoints.jsonl"
    self.clock = clock
    self.interval = float(interval_s)
    self.activation = activation
    self.k = 0

  def due(self) -> bool:
    c = self.clock
    ref = c.t_last_ckpt if c.t_last_ckpt is not None else c.t_launch
    return time.time() - ref >= self.interval

  def write(self, actor, iteration: int, env_steps: int, extra: dict | None = None) -> dict:
    now = time.time()
    p = policy_io.from_rsl_rl_actor(actor, activation=self.activation)
    path = self.ckpt_dir / f"policy_{self.k:04d}.pt"
    policy_io.save(p, path)
    c = self.clock
    row = {
      "k": self.k,
      "iteration": int(iteration),
      "env_steps": int(env_steps),
      "t_wall_s": now - c.t_launch,
      "t_train_s": (now - c.t_first_iter) if c.t_first_iter is not None else 0.0,
      "path": str(path.relative_to(self.run_dir)),
    }
    if extra:
      row.update(extra)
    with open(self.index_path, "a") as f:
      f.write(json.dumps(row) + "\n")
    c.t_last_ckpt = now
    self.k += 1
    return row


class IterLog:
  def __init__(self, run_dir: Path):
    self.f = open(Path(run_dir) / "iterations.jsonl", "a", buffering=1)

  def write(self, row: dict) -> None:
    self.f.write(json.dumps(row) + "\n")

  def close(self) -> None:
    self.f.close()


def write_run_meta(run_dir: Path, meta: dict) -> None:
  p = Path(run_dir) / "run.json"
  old = json.loads(p.read_text()) if p.exists() else {}
  old.update(meta)
  p.write_text(json.dumps(old, indent=2, default=str) + "\n")
