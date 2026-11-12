"""Stop a training run early once a separate machine has seen it pass the bar.

Runs from a workstation with ssh access to both machines:
  python scripts/live_stop.py --train-host A --eval-host B --remote-root /root/babysteps \
      --runs runs/official/stock_s0 runs/official/stock_s1 ...

Every minute: copy new checkpoints of the active run from the training machine to
the evaluation machine, evaluate them there in order, and once a checkpoint passes
and ``--buffer`` more checkpoints exist, create ``<run>/STOP`` on the training
machine. Training is never slowed (evaluation is on a different GPU) and the
official time-to-bar is recomputed afterwards by a full offline evaluation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path


def sh(host: str, cmd: str, check=True) -> str:
  r = subprocess.run(["ssh", "-o", "ConnectTimeout=15", host, cmd], capture_output=True, text=True)
  if check and r.returncode != 0:
    raise RuntimeError(f"{host}: {cmd}\n{r.stderr[-2000:]}")
  return r.stdout


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--train-host", required=True)
  ap.add_argument("--eval-host", required=True)
  ap.add_argument("--remote-root", default="/root/babysteps")
  ap.add_argument("--runs", nargs="+", required=True)
  ap.add_argument("--buffer", type=int, default=4, help="extra checkpoints to wait for after a pass")
  ap.add_argument("--procs", type=int, default=4)
  a = ap.parse_args()
  root = a.remote_root
  tmp = Path(tempfile.mkdtemp(prefix="live_stop_"))
  for run in a.runs:
    name = Path(run).name
    live = f"runs/live/{name}"
    next_k, pass_k = 0, None
    print(f"[{time.strftime('%H:%M:%S')}] watching {run}", flush=True)
    while True:
      exists = sh(a.train_host, f"test -f {root}/{run}/checkpoints.jsonl && echo y || echo n").strip() == "y"
      finished = exists and "finished_ts" in sh(a.train_host, f"cat {root}/{run}/run.json", check=False)
      if not exists:
        time.sleep(30)
        continue
      local = tmp / name
      local.mkdir(parents=True, exist_ok=True)
      subprocess.run(["rsync", "-az", f"{a.train_host}:{root}/{run}/", str(local) + "/",
                      "--include", "checkpoints/***", "--include", "checkpoints.jsonl", "--include", "run.json",
                      "--exclude", "*"], check=True)
      subprocess.run(["ssh", a.eval_host, f"mkdir -p {root}/{live}"], check=True)
      subprocess.run(["rsync", "-az", str(local) + "/", f"{a.eval_host}:{root}/{live}/"], check=True)
      n_ckpt = len((local / "checkpoints.jsonl").read_text().splitlines())
      if pass_k is None and next_k < n_ckpt:
        out = sh(a.eval_host, f"cd {root} && MUJOCO_GL=egl .venv/bin/python -m babysteps.eval.time_to_bar "
                              f"--run {live} --start-k {next_k} --procs {a.procs} 2>/dev/null | tail -n 40", check=False)
        try:
          summ = json.loads(out[out.index("{"):])
        except ValueError:
          summ = {"passed": False, "n_evaluated": 0}
        if summ.get("passed"):
          pass_k = summ["first_pass_k"]
          print(f"[{time.strftime('%H:%M:%S')}] {name}: live pass at k={pass_k} "
                f"({summ['time_to_bar_s']/60:.2f} min)", flush=True)
        else:
          next_k += summ.get("n_evaluated", 0)
      if pass_k is not None and n_ckpt >= pass_k + 1 + a.buffer:
        sh(a.train_host, f"touch {root}/{run}/STOP")
        print(f"[{time.strftime('%H:%M:%S')}] {name}: STOP sent", flush=True)
      if finished:
        print(f"[{time.strftime('%H:%M:%S')}] {name}: finished (live pass k={pass_k})", flush=True)
        break
      time.sleep(45)


if __name__ == "__main__":
  main()
