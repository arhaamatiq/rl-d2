"""Build the ablation table (markdown) and ledger rows from run directories.

  python scripts/ablation_table.py results/ablation results/official [--ledger]

Rows are cumulative: each adds one change to the row above. Row 0 (stock) and the last
row (track_a) are the official 5-seed results; middle rows use seeds 0-2.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from babysteps import ledger as L  # noqa: E402

ROWS = [
  ("stock", "stock mjlab PPO (T0)"),
  ("graph_stock", "+ whole rollout and PPO update as CUDA graphs (same hyperparameters)"),
  ("abl_kl02", "+ KL target 0.01 -> 0.02"),
  ("abl_kl02_sym", "+ left-right symmetry augmentation"),
  ("abl_kl02_sym_e8", "+ 8 PPO epochs (was 5)"),
  ("abl_kl02_sym_e8_bf16", "+ bf16 autocast in the update"),
  ("track_a", "+ yaw-rate tracking weight 2 -> 4 (= Track A)"),
]


def collect(dirs: list[Path], recipe: str) -> list[dict]:
  out = []
  for root in dirs:
    for d in sorted(root.glob(f"{recipe}_s[0-9]")):
      t = d / "time_to_bar.json"
      if not t.exists():
        continue
      tt = json.loads(t.read_text())
      meta = json.loads((d / "run.json").read_text())
      it = [json.loads(x) for x in (d / "iterations.jsonl").read_text().splitlines()[50:] if x]
      spi = statistics.median(
        (x["iter_s"] if "iter_s" in x else x["collect_s"] + x["learn_s"]) for x in it
      ) if it else float("nan")
      out.append({
        "seed": meta["seed"], "passed": tt["passed"], "ttb": tt.get("time_to_bar_s", float("nan")) / 60,
        "iters": tt.get("iteration"), "spi": spi, "startup": (meta.get("clock") or {}).get("first_iter_start"),
        "host": meta.get("host"), "dir": str(d),
      })
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("dirs", nargs="+", type=Path)
  ap.add_argument("--ledger", action="store_true")
  a = ap.parse_args()
  lines = [
    "| # | change (cumulative) | seeds | time-to-bar, median [IQR] (min) | iterations to bar | s / iteration | startup (s) |",
    "|---|---|---|---|---|---|---|",
  ]
  prev = None
  for i, (recipe, label) in enumerate(ROWS):
    rs = collect(a.dirs, recipe)
    if not rs:
      lines.append(f"| {i} | {label} | - | not run | | | |")
      continue
    ok = [r for r in rs if r["passed"]]
    seeds = ",".join(str(r["seed"]) for r in sorted(rs, key=lambda r: r["seed"]))
    if ok:
      q1, med, q3 = L.quantiles([r["ttb"] for r in ok])
      cell = f"**{med:.2f}** [{q1:.2f}, {q3:.2f}]"
      if len(ok) < len(rs):
        cell += f" ({len(rs) - len(ok)} did not pass)"
    else:
      med, cell = float("nan"), "no pass"
    its = statistics.median([r["iters"] for r in ok]) if ok else float("nan")
    spi = statistics.median([r["spi"] for r in rs])
    su = statistics.median([r["startup"] for r in rs if r["startup"]])
    delta = f" ({prev / med:.2f}x vs row {i - 1})" if prev and ok else ""
    lines.append(f"| {i} | {label} | {seeds} | {cell}{delta} | {its:.0f} | {spi:.3f} | {su:.0f} |")
    if a.ledger and ok and recipe not in ("stock", "track_a"):
      L.add({
        "id": f"AB{i}", "metric": f"ablation row {i}: {label} - time-to-bar", "value": round(med, 2), "unit": "min",
        "seeds": seeds, "median_iqr": f"{med:.2f} [{q1:.2f}, {q3:.2f}]",
        "command": f"scripts/run_queue.sh runs/ablation {recipe}:<seed>:<budget>",
        "gpu": "RTX 4090 + Ryzen 9 7950X", "artifact": f"results/ablation/{recipe}_s*/",
        "per_seed": [round(r["ttb"], 2) for r in sorted(rs, key=lambda r: r["seed"])],
        "iters_median": its, "s_per_iter_median": round(spi, 3),
      })
    prev = med if ok else prev
  table = "\n".join(lines)
  print(table)
  if a.ledger:
    L.render()


if __name__ == "__main__":
  main()
