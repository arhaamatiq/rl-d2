"""Median / IQR of time-to-bar over run directories, grouped by recipe.

  python scripts/summarize.py runs/official/*/ [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rl_d2.ledger import quantiles


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("runs", nargs="+")
  ap.add_argument("--json", default=None)
  a = ap.parse_args()
  groups: dict[str, list] = defaultdict(list)
  for d in a.runs:
    d = Path(d)
    if not (d / "run.json").exists():
      continue
    meta = json.loads((d / "run.json").read_text())
    ttb = json.loads((d / "time_to_bar.json").read_text()) if (d / "time_to_bar.json").exists() else {}
    key = meta["recipe"] + ("" if not meta.get("overrides") else " " + json.dumps(meta["overrides"]))
    groups[key].append({
      "seed": meta["seed"], "run": str(d), "passed": ttb.get("passed", False),
      "ttb_min": ttb.get("time_to_bar_s", float("nan")) / 60,
      "ttb_train_min": ttb.get("time_to_bar_train_s", float("nan")) / 60,
      "iteration": ttb.get("iteration"), "env_steps": ttb.get("env_steps"),
      "startup_s": meta.get("clock", {}).get("first_iter_start"),
      "gpu_util": (meta.get("gpu_monitor_train") or {}).get("gpu_util_mean"),
    })
  out = {}
  for k, rs in groups.items():
    ok = [r for r in rs if r["passed"]]
    row = {"n": len(rs), "passed": len(ok), "runs": sorted(rs, key=lambda r: r["seed"])}
    if ok:
      for f in ("ttb_min", "ttb_train_min"):
        q1, med, q3 = quantiles([r[f] for r in ok])
        row[f] = {"median": med, "q1": q1, "q3": q3}
    out[k] = row
    t = row.get("ttb_min")
    print(f"{k}: {len(ok)}/{len(rs)} passed" + (f", median {t['median']:.2f} min [IQR {t['q1']:.2f}-{t['q3']:.2f}]" if t else ""))
  if a.json:
    Path(a.json).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
  main()
