"""Training entry point used by every official run.

  scripts/train.sh <recipe> <seed> <run_dir> [budget_min]

The wrapper exports BS_LAUNCH_TS before the interpreter starts, so imports,
kernel compilation, env construction and graph capture are all inside the clock.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from babysteps.bar import load_bar
from babysteps.train.harness import CheckpointWriter, IterLog, RunClock, launch_timestamp, write_run_meta


def _git_rev() -> str:
  try:
    root = Path(__file__).resolve().parents[3]
    rev = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True).strip()
    dirty = subprocess.call(["git", "-C", str(root), "diff", "--quiet"]) != 0
    return rev + ("-dirty" if dirty else "")
  except Exception:
    return os.environ.get("BS_COMMIT", "unknown")


def _parse_value(v: str):
  import ast

  try:
    return ast.literal_eval(v)
  except Exception:
    return v


def apply_overrides(recipe, sets: list[str]) -> dict:
  """Dotted-path overrides on env / agent configs or recipe options."""
  done = {}
  roots = {"env": recipe.env_cfg, "agent": recipe.agent_cfg}
  for item in sets:
    path, val = item.split("=", 1)
    val = _parse_value(val)
    head, *rest = path.split(".")
    if head == "opt":
      recipe.options[".".join(rest)] = val
    else:
      obj = roots[head]
      for part in rest[:-1]:
        obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
      if isinstance(obj, dict):
        obj[rest[-1]] = val
      else:
        if not hasattr(obj, rest[-1]):
          raise AttributeError(f"{path}: no field {rest[-1]!r}")
        setattr(obj, rest[-1], val)
    done[path] = val
  return done


def main(argv: list[str] | None = None) -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--recipe", default="stock")
  ap.add_argument("--seed", type=int, required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--budget-min", type=float, default=90.0)
  ap.add_argument("--max-iterations", type=int, default=None)
  ap.add_argument("--profile-iters", type=str, default=None, help="start:stop, cudaProfilerApi range")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--set", action="append", default=[], metavar="PATH=VALUE",
                  help="override a config field, e.g. env.scene.num_envs=8192 or agent.algorithm.desired_kl=0.02")
  args = ap.parse_args(argv)

  t_launch = launch_timestamp()
  clock = RunClock(t_launch=t_launch)
  clock.mark("python_main")
  run_dir: Path = args.out.resolve()
  run_dir.mkdir(parents=True, exist_ok=True)
  bar = load_bar()

  from babysteps.systems.gpu_monitor import GpuMonitor

  mon = GpuMonitor().start()

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.utils.torch import configure_torch_backends

  from babysteps.systems import nvtx
  from babysteps.train.recipes import get_recipe

  clock.mark("imports_done")
  recipe = get_recipe(args.recipe)
  env_cfg, agent_cfg = recipe.env_cfg, recipe.agent_cfg
  overrides = apply_overrides(recipe, args.set)
  configure_torch_backends()
  agent_cfg.seed = args.seed
  env_cfg.seed = args.seed
  if args.max_iterations is not None:
    agent_cfg.max_iterations = args.max_iterations
  agent_cfg.run_name = run_dir.name

  write_run_meta(run_dir, {
    "recipe": recipe.name,
    "seed": args.seed,
    "budget_min": args.budget_min,
    "launch_ts": t_launch,
    "commit": os.environ.get("BS_COMMIT") or _git_rev(),
    "bar_sha256": bar.sha256,
    "host": socket.gethostname(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "torch": torch.__version__,
    "python": platform.python_version(),
    "argv": sys.argv,
    "options": recipe.options,
    "overrides": overrides,
  })

  prof = None
  if args.profile_iters:
    a, b = args.profile_iters.split(":")
    prof = (int(a), int(b))

  if recipe.runner == "stock":
    from babysteps.train.stock_runner import TimedVelocityRunner

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    nvtx.instrument_env(env)
    clock.mark("env_built")

    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    tb_dir = run_dir / "tb"
    runner = TimedVelocityRunner(venv, asdict(agent_cfg), str(tb_dir), args.device)
    ckpt = CheckpointWriter(run_dir, clock, bar.raw["checkpoint_interval_s"], activation=agent_cfg.actor.activation)
    iterlog = IterLog(run_dir)
    runner.attach_harness(clock, ckpt, args.budget_min * 60.0, iterlog, prof)
    clock.mark("runner_built")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    iterlog.close()
  else:
    from babysteps.train.fast_runner import run_fast

    trainer = run_fast(recipe, run_dir, clock, bar, args, prof)
    env = trainer.genv.env

  clock.mark("done")
  t_first = clock.events.get("first_iter_start")
  mon.stop()
  write_run_meta(run_dir, {
    "clock": clock.events,
    "gpu_monitor_all": mon.summary(),
    "gpu_monitor_train": mon.summary(t_from=t_launch + t_first + 30.0) if t_first else None,
    "finished_ts": time.time(),
  })
  print(json.dumps({"run_dir": str(run_dir), "clock": clock.events}, indent=1))
  env.close()


if __name__ == "__main__":
  main()
