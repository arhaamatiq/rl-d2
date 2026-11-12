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


def evaluate_run(
  run_dir: Path,
  bar_path: Path | None = None,
  all_ckpts: bool = False,
  sim2sim_all: bool = False,
  device: str = "cuda:0",
  workers: int = 8,
  start_k: int = 0,
  every: int = 1,
) -> dict:
  from babysteps.eval.gpu_eval import EvalEnv, run_gpu_eval
  from babysteps.eval.sim2sim import run_sim2sim

  bar = load_bar(bar_path)
  thr = bar.thresholds
  calibrating = any(v is None for v in thr.values())
  ckpts = [c for c in read_checkpoints(run_dir) if c["k"] >= start_k]
  if every > 1:
    ckpts = [c for c in ckpts if c["k"] % every == 0 or c.get("final")]
  ev = EvalEnv.create(bar, device=device)
  out_path = run_dir / ("eval_all.jsonl" if all_ckpts else "eval.jsonl")
  first = None
  t0 = time.time()
  with open(out_path, "a") as f:
    for c in ckpts:
      pol = policy_io.load(run_dir / c["path"])
      m = run_gpu_eval(ev, pol)
      m.pop("frames", None)
      row = {"k": c["k"], "t_wall_s": c["t_wall_s"], "t_train_s": c["t_train_s"],
             "iteration": c["iteration"], "env_steps": c["env_steps"], **m}
      passed = False
      if not calibrating:
        g = gpu_pass(m, thr)
        row["gpu_checks"] = g
        if all(g.values()) or sim2sim_all:
          s = run_sim2sim(bar, pol, workers=workers)
          row.update(s)
          row["sim2sim_ok"] = sim2sim_pass(s, thr)
          passed = all(g.values()) and row["sim2sim_ok"]
      elif sim2sim_all:
        row.update(run_sim2sim(bar, pol, workers=workers))
      row["passed"] = passed
      f.write(json.dumps(row) + "\n")
      f.flush()
      print(
        f"k={c['k']:4d} t={c['t_wall_s']/60:6.2f}min it={c['iteration']:6d} "
        f"lin={m['lin_err']:.3f} yaw={m['yaw_err']:.3f} ff={m['fall_free']:.3f} "
        f"push={m.get('push_success', float('nan')):.3f}"
        + (f" s2s={row['sim2sim_success']:.3f}" if "sim2sim_success" in row else "")
        + (" PASS" if passed else ""),
        flush=True,
      )
      if passed and first is None:
        first = row
        if not all_ckpts:
          break
  summary = {
    "run_dir": str(run_dir),
    "bar_version": bar.version,
    "bar_sha256": bar.sha256,
    "frozen": bar.frozen,
    "n_evaluated": len(ckpts),
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
      "metrics": {k: first[k] for k in ("lin_err", "yaw_err", "fall_free", "push_success",
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
  a = ap.parse_args(argv)
  s = evaluate_run(a.run, a.bar, a.all, a.sim2sim_all, a.device, a.workers, a.start_k, a.every)
  print(json.dumps(s, indent=2))


if __name__ == "__main__":
  main()
