"""Offline evaluation of a run: find the first checkpoint that passes the bar.

  bs-eval --run runs/<name>                 # stop at the first pass (official)
  bs-eval --run runs/<name> --all           # every checkpoint (calibration curves)

Stages per checkpoint, in order, each gating the next:
  1. GPU: tracking error (lin, yaw) and fall-free rate over the held-out commands
  2. GPU: push recovery at the fixed push strength (same batched rollout)
  3. CPU: sim-to-sim replay success rate
Time-to-bar is the checkpoint's ``t_wall_s`` (launch -> checkpoint written).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from babysteps import policy as policy_io
from babysteps.bar import load_bar


def gpu_pass(m: dict, thr: dict) -> dict:
  c = {
    "lin_err": m["lin_err"] <= thr["lin_err_max"],
    "yaw_err": m["yaw_err"] <= thr["yaw_err_max"],
    "fall_free": m["fall_free"] >= thr["fall_free_min"],
    "push": m.get("push_success", 0.0) >= thr["push_success_min"],
  }
  return c


def sim2sim_pass(m: dict, thr: dict) -> bool:
  return m["sim2sim_success"] >= thr["sim2sim_success_min"]


def read_checkpoints(run_dir: Path) -> list[dict]:
  rows = [json.loads(line) for line in (run_dir / "checkpoints.jsonl").read_text().splitlines() if line]
  return sorted(rows, key=lambda r: r["k"])


_EV = None


def _eval_one(args) -> dict:
  """Evaluate one checkpoint (runs in a worker process that keeps its EvalEnv)."""
  global _EV
  run_dir, bar_path, c, calibrating, sim2sim_all, device, s2s_workers = args
  from babysteps.eval.gpu_eval import EvalEnv, run_gpu_eval
  from babysteps.eval.sim2sim import run_sim2sim

  bar = load_bar(bar_path)
  thr = bar.thresholds
  if _EV is None:
    _EV = EvalEnv.create(bar, device=device)
  pol = policy_io.load(Path(run_dir) / c["path"])
  m = run_gpu_eval(_EV, pol)
  m.pop("frames", None)
  row = {"k": c["k"], "t_wall_s": c["t_wall_s"], "t_train_s": c["t_train_s"],
         "iteration": c["iteration"], "env_steps": c["env_steps"], **m}
  passed = False
  if not calibrating:
    g = gpu_pass(m, thr)
    row["gpu_checks"] = g
    if all(g.values()) or sim2sim_all:
      s = run_sim2sim(bar, pol, workers=s2s_workers)
      row.update(s)
      row["sim2sim_ok"] = sim2sim_pass(s, thr)
      passed = all(g.values()) and row["sim2sim_ok"]
  elif sim2sim_all:
    row.update(run_sim2sim(bar, pol, workers=s2s_workers))
  row["passed"] = passed
  return row


def _fmt_row(row: dict) -> str:
  return (
    f"k={row['k']:4d} t={row['t_wall_s']/60:6.2f}min it={row['iteration']:6d} "
    f"lin={row['lin_err']:.3f} yaw={row['yaw_err']:.3f} ff={row['fall_free']:.3f} "
    f"push={row.get('push_success', float('nan')):.3f}"
    + (f" s2s={row['sim2sim_success']:.3f}" if "sim2sim_success" in row else "")
    + (" PASS" if row["passed"] else "")
  )


def evaluate_run(
  run_dir: Path,
  bar_path: Path | None = None,
  all_ckpts: bool = False,
  sim2sim_all: bool = False,
  device: str = "cuda:0",
  workers: int = 8,
  start_k: int = 0,
  every: int = 1,
  procs: int = 1,
) -> dict:
  """Checkpoints are evaluated in order, in chunks of ``procs`` processes sharing
  the GPU (the eval env is host-bound, so this multiplies throughput). The first
  passing checkpoint is the smallest passing k; every earlier k has been
  evaluated, exactly as in a serial scan."""
  import multiprocessing as mp
  from concurrent.futures import ProcessPoolExecutor

  bar = load_bar(bar_path)
  thr = bar.thresholds
  calibrating = any(v is None for v in thr.values())
  ckpts = [c for c in read_checkpoints(run_dir) if c["k"] >= start_k]
  if every > 1:
    ckpts = [c for c in ckpts if c["k"] % every == 0 or c.get("final")]
  out_path = run_dir / ("eval_all.jsonl" if all_ckpts else "eval.jsonl")
  s2s_workers = max(1, workers // procs)
  jobs = [(str(run_dir), str(bar.path), c, calibrating, sim2sim_all, device, s2s_workers) for c in ckpts]
  first = None
  t0 = time.time()
  n_done = 0
  pool = ProcessPoolExecutor(max_workers=procs, mp_context=mp.get_context("spawn")) if procs > 1 else None
  try:
    with open(out_path, "a") as f:
      for i in range(0, len(jobs), procs):
        chunk = jobs[i : i + procs]
        rows = list(pool.map(_eval_one, chunk)) if pool else [_eval_one(j) for j in chunk]
        for row in rows:
          f.write(json.dumps(row) + "\n")
          print(_fmt_row(row), flush=True)
        f.flush()
        n_done += len(rows)
        passing = [r for r in rows if r["passed"]]
        if passing and first is None:
          first = min(passing, key=lambda r: r["k"])
          if not all_ckpts:
            break
  finally:
    if pool:
      pool.shutdown()
  summary = {
    "run_dir": str(run_dir),
    "bar_version": bar.version,
    "bar_sha256": bar.sha256,
    "frozen": bar.frozen,
    "n_evaluated": n_done,
    "eval_wall_s": time.time() - t0,
    "passed": first is not None,
  }
  if first is not None:
    summary.update({
      "first_pass_k": first["k"],
      "time_to_bar_s": first["t_wall_s"],
      "time_to_bar_train_s": first["t_train_s"],
      "iteration": first["iteration"],
      "env_steps": first["env_steps"],
      "metrics": {k: first.get(k) for k in ("lin_err", "yaw_err", "fall_free", "push_success",
                                            "sim2sim_success", "sim2sim_lin_err")},
    })
  if not calibrating:
    (run_dir / "time_to_bar.json").write_text(json.dumps(summary, indent=2) + "\n")
  return summary


def main(argv: list[str] | None = None) -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", type=Path, required=True)
  ap.add_argument("--bar", type=Path, default=None)
  ap.add_argument("--all", action="store_true")
  ap.add_argument("--sim2sim-all", action="store_true")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--workers", type=int, default=8)
  ap.add_argument("--start-k", type=int, default=0)
  ap.add_argument("--every", type=int, default=1, help="calibration only: evaluate every n-th checkpoint")
  ap.add_argument("--procs", type=int, default=1, help="evaluation processes sharing the GPU")
  a = ap.parse_args(argv)
  s = evaluate_run(a.run, a.bar, a.all, a.sim2sim_all, a.device, a.workers, a.start_k, a.every, a.procs)
  print(json.dumps(s, indent=2))


if __name__ == "__main__":
  main()
