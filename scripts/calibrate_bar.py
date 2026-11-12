"""Apply the calibration rule (DECISIONS.md D6) and freeze the bar.

  python scripts/calibrate_bar.py runs/calib/stock_s100 runs/calib/stock_s101 runs/calib/stock_s102 --freeze

Reference policy per calibration seed = its final checkpoint. Thresholds:
  lin/yaw error max      = 1.25 x worst reference
  fall_free_min          = 0.98 if every reference is fall-free, else worst - 0.02
  push strength          = largest candidate where every reference survives >= 95%
  push_success_min       = 0.90
  sim2sim_lin_err_max    = 1.5 x lin_err_max
  sim2sim_success_min    = 0.875, provided every reference scores >= 15/16
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from babysteps import policy as policy_io
from babysteps.bar import Bar, load_bar

PUSH_CANDIDATES = [0.5, 0.75, 1.0, 1.25]


def _ceil(x: float, step: float = 0.005) -> float:
  return math.ceil(x / step - 1e-9) * step


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("runs", nargs="+", type=Path)
  ap.add_argument("--freeze", action="store_true")
  ap.add_argument("--device", default="cuda:0")
  a = ap.parse_args()

  from babysteps.eval.gpu_eval import EvalEnv, run_gpu_eval
  from babysteps.eval.sim2sim import run_sim2sim

  base = load_bar()
  refs = []
  for run in a.runs:
    rows = [json.loads(x) for x in (run / "checkpoints.jsonl").read_text().splitlines() if x]
    final = max(rows, key=lambda r: r["k"])
    pol = policy_io.load(run / final["path"])
    per_push = {}
    track = None
    for dv in PUSH_CANDIDATES:
      raw = copy.deepcopy(base.raw)
      raw["push"]["delta_v_mps"] = dv
      bar = Bar(raw=raw, sha256="calibration", path=base.path)
      ev = EvalEnv.create(bar, device=a.device)
      m = run_gpu_eval(ev, pol)
      m.pop("frames", None)
      per_push[str(dv)] = m["push_success"]
      if track is None:
        track = {k: m[k] for k in ("lin_err", "yaw_err", "fall_free")}
      ev.env.close()
      del ev
    s = run_sim2sim(base, pol, workers=8)
    ref = {
      "run": str(run), "k": final["k"], "iteration": final["iteration"], "t_wall_s": final["t_wall_s"],
      **track, "push_success": per_push,
      "sim2sim_fall_free": s["sim2sim_fall_free"], "sim2sim_lin_err": s["sim2sim_lin_err"],
      "sim2sim_lin_errs": [e["lin_err"] for e in s["sim2sim_episodes"]],
      "sim2sim_falls": [e["fell_at_step"] for e in s["sim2sim_episodes"]],
    }
    print(json.dumps(ref), flush=True)
    refs.append(ref)

  lin_max = _ceil(1.25 * max(r["lin_err"] for r in refs))
  yaw_max = _ceil(1.25 * max(r["yaw_err"] for r in refs))
  worst_ff = min(r["fall_free"] for r in refs)
  ff_min = 0.98 if worst_ff >= 1.0 else round(worst_ff - 0.02, 4)
  ok_dv = [dv for dv in PUSH_CANDIDATES if all(r["push_success"][str(dv)] >= 0.95 for r in refs)]
  push_dv = max(ok_dv) if ok_dv else None
  s2s_lin = _ceil(1.5 * lin_max)
  s2s_scores = [sum((f is None) and (e <= s2s_lin) for f, e in zip(r["sim2sim_falls"], r["sim2sim_lin_errs"], strict=True))
                for r in refs]
  thresholds = {
    "lin_err_max": lin_max,
    "yaw_err_max": yaw_max,
    "fall_free_min": ff_min,
    "push_success_min": 0.90,
    "sim2sim_lin_err_max": s2s_lin,
    "sim2sim_success_min": 0.875,
  }
  result = {"thresholds": thresholds, "push_delta_v_mps": push_dv, "sim2sim_scores_of_16": s2s_scores}
  print(json.dumps(result, indent=2))
  if push_dv is None:
    raise SystemExit("no candidate push strength passes the rule; stop and review")
  if min(s2s_scores) < 15:
    raise SystemExit("a reference policy scores < 15/16 on sim-to-sim; stop and review")
  if a.freeze:
    raw = copy.deepcopy(base.raw)
    raw["thresholds"] = thresholds
    raw["push"]["delta_v_mps"] = push_dv
    raw["frozen"] = True
    raw["calibration"] = {
      "rule": "DECISIONS.md D6",
      "seeds": [json.loads((Path(r["run"]) / "run.json").read_text())["seed"] for r in refs],
      "reference": "final checkpoint of a 120-minute stock run (3 runs sharing one GPU)",
      "references": [{k: v for k, v in r.items() if k not in ("sim2sim_lin_errs", "sim2sim_falls")} for r in refs],
      "frozen_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
    }
    base.path.write_text(json.dumps(raw, indent=2) + "\n")
    print("frozen:", load_bar(base.path).sha256)


if __name__ == "__main__":
  main()
