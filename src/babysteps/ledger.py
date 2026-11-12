"""The metrics ledger: the only source of numbers for the write-up.

``results/ledger.jsonl`` is append-only; ``LEDGER.md`` is rendered from it.

  bs-ledger add --id T0 --metric time_to_bar --value 1234.5 --unit s ...
  bs-ledger runs --id T0 --metric time_to_bar --runs runs/official/stock_s{0..4} ...
  bs-ledger render
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from babysteps.bar import REPO_ROOT

LEDGER = REPO_ROOT / "results" / "ledger.jsonl"
LEDGER_MD = REPO_ROOT / "LEDGER.md"
COLUMNS = ["id", "metric", "value", "unit", "seeds", "median_iqr", "command", "commit", "gpu", "date", "artifact"]


def _commit() -> str:
  try:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
  except Exception:
    return "unknown"


def quantiles(xs: list[float]) -> tuple[float, float, float]:
  xs = sorted(xs)
  if len(xs) == 1:
    return xs[0], xs[0], xs[0]
  q = statistics.quantiles(xs, n=4, method="inclusive")
  return q[0], statistics.median(xs), q[2]


def add(row: dict) -> dict:
  row = {k: row.get(k, "") for k in COLUMNS} | {k: v for k, v in row.items() if k not in COLUMNS}
  row["date"] = row.get("date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
  row["commit"] = row.get("commit") or _commit()
  LEDGER.parent.mkdir(parents=True, exist_ok=True)
  with open(LEDGER, "a") as f:
    f.write(json.dumps(row) + "\n")
  return row


def rows() -> list[dict]:
  if not LEDGER.exists():
    return []
  return [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]


def _fmt(v) -> str:
  if isinstance(v, float):
    return f"{v:.4g}" if abs(v) < 1000 else f"{v:.1f}"
  return str(v)


def render() -> str:
  out = [
    "# Ledger",
    "",
    "Every number in the README and write-up comes from this table. Rendered from",
    "`results/ledger.jsonl` by `bs-ledger render`; edit the jsonl, never this file.",
    "",
    "| " + " | ".join(COLUMNS) + " |",
    "|" + "---|" * len(COLUMNS),
  ]
  for r in rows():
    out.append("| " + " | ".join(_fmt(r.get(c, "")).replace("|", "\\|") for c in COLUMNS) + " |")
  text = "\n".join(out) + "\n"
  LEDGER_MD.write_text(text)
  return text


def from_runs(run_dirs: list[Path], key: str) -> dict:
  vals, seeds, failed = [], [], []
  for d in run_dirs:
    p = Path(d) / "time_to_bar.json"
    meta = json.loads((Path(d) / "run.json").read_text())
    seeds.append(meta["seed"])
    s = json.loads(p.read_text()) if p.exists() else {"passed": False}
    if s.get("passed"):
      vals.append(float(s[key]))
    else:
      failed.append(meta["seed"])
  return {"values": vals, "seeds": seeds, "failed": failed}


def main(argv: list[str] | None = None) -> None:
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest="cmd", required=True)
  a_add = sub.add_parser("add")
  for c in COLUMNS:
    a_add.add_argument(f"--{c.replace('_', '-')}", default="")
  a_runs = sub.add_parser("runs", help="median/IQR of a time_to_bar.json field over run dirs")
  a_runs.add_argument("--id", required=True)
  a_runs.add_argument("--metric", required=True)
  a_runs.add_argument("--field", default="time_to_bar_s")
  a_runs.add_argument("--scale", type=float, default=1.0 / 60.0, help="default: seconds -> minutes")
  a_runs.add_argument("--unit", default="min")
  a_runs.add_argument("--command", default="")
  a_runs.add_argument("--gpu", default="RTX 4090")
  a_runs.add_argument("--artifact", default="")
  a_runs.add_argument("--runs", nargs="+", required=True)
  sub.add_parser("render")
  a = ap.parse_args(argv)
  if a.cmd == "add":
    r = add({c: getattr(a, c) for c in COLUMNS})
    print(json.dumps(r))
  elif a.cmd == "runs":
    res = from_runs([Path(x) for x in a.runs], a.field)
    vals = [v * a.scale for v in res["values"]]
    if res["failed"]:
      raise SystemExit(f"seeds without a passing checkpoint: {res['failed']}")
    q1, med, q3 = quantiles(vals)
    r = add({
      "id": a.id, "metric": a.metric, "value": med, "unit": a.unit,
      "seeds": ",".join(str(s) for s in res["seeds"]),
      "median_iqr": f"{med:.2f} [{q1:.2f}, {q3:.2f}]",
      "command": a.command, "gpu": a.gpu, "artifact": a.artifact or ",".join(a.runs),
      "per_seed": [round(v, 3) for v in vals],
    })
    print(json.dumps(r))
  render()


if __name__ == "__main__":
  main()
