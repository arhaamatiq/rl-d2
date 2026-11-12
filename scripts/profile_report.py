"""Summarize an Nsight Systems trace of a training run by NVTX phase.

  nsys export --type sqlite -o prof.sqlite prof.nsys-rep
  python scripts/profile_report.py prof.sqlite [--json out.json]

Reports, over the profiled iterations:
  * host wall time per phase (NVTX ranges on the main thread)
  * GPU kernel time per phase (each kernel attributed to the innermost NVTX range
    open on the launching thread when its launch / graph-launch API call was made)
  * kernel coverage = |union of kernel intervals| / window length
    (fraction of wall time in which at least one kernel is running)
  * host<->device copies and synchronizing API calls
"""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
from collections import defaultdict

PHASES = [
  ("simulation", ("env.physics", "env.forward", "env.sense")),
  ("env_managers", ("env.managers", "env.reset_idx")),
  ("policy_inference", ("rollout.policy",)),
  ("rollout_storage", ("rollout.storage",)),
  ("gae", ("gae",)),
  ("backprop", ("update.backward",)),
  ("optimizer_step", ("update.optim", "update.clip_grad")),
  ("update_forward_loss", ("update",)),
  ("logging_ckpt", ("log", "ckpt")),
  ("host_sync_item", ("sync.item",)),
]


def phase_of(name: str) -> str:
  for ph, prefixes in PHASES:
    for p in prefixes:
      if name == p or name.startswith(p + "."):
        return ph
  if name.startswith("rollout"):
    return "rollout_other"
  if name.startswith("iter"):
    return "loop_other"
  return "other"


def union_length(intervals: list[tuple[int, int]]) -> int:
  total, cur_s, cur_e = 0, None, None
  for s, e in sorted(intervals):
    if cur_e is None or s > cur_e:
      if cur_e is not None:
        total += cur_e - cur_s
      cur_s, cur_e = s, e
    else:
      cur_e = max(cur_e, e)
  if cur_e is not None:
    total += cur_e - cur_s
  return total


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("sqlite")
  ap.add_argument("--json", default=None)
  a = ap.parse_args()
  db = sqlite3.connect(a.sqlite)
  q = lambda s: db.execute(s).fetchall()  # noqa: E731
  tables = {r[0] for r in q("select name from sqlite_master where type='table'")}
  strings = dict(q("select id, value from StringIds"))

  nv_cols = [r[1] for r in q("pragma table_info(NVTX_EVENTS)")]
  text_expr = "coalesce(text, '')" if "textId" not in nv_cols else "coalesce(text, '')"
  rows = q(f"select start, end, {text_expr}, textId, globalTid from NVTX_EVENTS where end is not null")
  ranges = []
  for s, e, t, tid_text, gtid in rows:
    name = t or strings.get(tid_text, "")
    if name:
      ranges.append((s, e, name, gtid))
  iters = [r for r in ranges if r[2].startswith("iter ")]
  if not iters:
    raise SystemExit("no 'iter' NVTX ranges in capture")
  w0, w1 = min(r[0] for r in iters), max(r[1] for r in iters)
  win = w1 - w0
  main_tid = iters[0][3]

  # host time per phase: exclusive time of each innermost range on the main thread
  mt = sorted([r for r in ranges if r[3] == main_tid and r[0] >= w0 and r[1] <= w1], key=lambda r: (r[0], -r[1]))
  host = defaultdict(int)
  host_raw = defaultdict(int)
  events = []
  for s, e, n, _ in mt:
    events.append((s, 1, n, e))
  # exclusive-time sweep
  stack: list[tuple[str, int]] = []
  points = sorted([(s, 0, n) for s, e, n, _ in mt] + [(e, 1, n) for s, e, n, _ in mt], key=lambda x: (x[0], -x[1]))
  last_t = w0
  for t, kind, n in points:
    if stack:
      host[phase_of(stack[-1][0])] += t - last_t
      host_raw[stack[-1][0] if not stack[-1][0].startswith("iter ") else "iter"] += t - last_t
    else:
      host["outside_ranges"] += t - last_t
    last_t = t
    if kind == 0:
      stack.append((n, t))
    else:
      for i in range(len(stack) - 1, -1, -1):
        if stack[i][0] == n:
          stack.pop(i)
          break

  # kernels -> launching API call -> innermost NVTX range on that thread
  kern = q(
    "select start, end, correlationId from CUPTI_ACTIVITY_KIND_KERNEL "
    f"where start >= {w0} and start <= {w1}"
  )
  n_kernels = len(kern)
  n_graphs = 0
  if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables:
    # CUDA graphs traced at graph granularity: one interval per graph launch.
    graphs = q(
      "select start, end, correlationId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE "
      f"where start >= {w0} and start <= {w1}"
    )
    n_graphs = len(graphs)
    kern = kern + graphs
  rt = q("select start, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME")
  corr = {c: (s, g) for s, c, g in rt}
  by_tid: dict[int, list] = defaultdict(list)
  for s, e, n, g in ranges:
    by_tid[g].append((s, e, n))
  for g in by_tid:
    by_tid[g].sort()
  starts = {g: [r[0] for r in v] for g, v in by_tid.items()}

  def innermost(g: int, t: int) -> str:
    lst = by_tid.get(g, [])
    i = bisect.bisect_right(starts.get(g, []), t)
    best, best_len = "unannotated", None
    for j in range(i - 1, max(-1, i - 400), -1):
      s, e, n = lst[j]
      if s <= t <= e and (best_len is None or e - s < best_len):
        best, best_len = n, e - s
    return best

  gpu = defaultdict(int)
  intervals = []
  for s, e, c in kern:
    intervals.append((s, min(e, w1)))
    api = corr.get(c)
    ph = phase_of(innermost(api[1], api[0])) if api else "unattributed"
    gpu[ph] += e - s
  coverage = union_length(intervals) / win

  memcpy = defaultdict(lambda: [0, 0, 0])
  if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
    kinds = {1: "HtoD", 2: "DtoH", 8: "DtoD"}
    for s, e, b, k in q(
      f"select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where start >= {w0} and start <= {w1}"
    ):
      m = memcpy[kinds.get(k, str(k))]
      m[0] += 1
      m[1] += e - s
      m[2] += b
  sync_calls = 0
  for (nid,) in q(
    f"select nameId from CUPTI_ACTIVITY_KIND_RUNTIME where start >= {w0} and start <= {w1}"
  ):
    nm = strings.get(nid, "")
    if "Synchronize" in nm or nm.startswith("cudaMemcpy") and "Async" not in nm:
      sync_calls += 1

  n_it = len(iters)
  out = {
    "iterations": n_it,
    "window_s": win / 1e9,
    "s_per_iter": win / 1e9 / n_it,
    "kernel_coverage": coverage,
    "host_ms_per_iter": {k: v / 1e6 / n_it for k, v in sorted(host.items(), key=lambda x: -x[1])},
    "host_ms_per_iter_by_range": {k: v / 1e6 / n_it for k, v in sorted(host_raw.items(), key=lambda x: -x[1])[:25]},
    "gpu_kernel_ms_per_iter": {k: v / 1e6 / n_it for k, v in sorted(gpu.items(), key=lambda x: -x[1])},
    "kernels_per_iter": n_kernels / n_it,
    "graph_launches_per_iter": n_graphs / n_it,
    "memcpy_per_iter": {k: {"count": v[0] / n_it, "ms": v[1] / 1e6 / n_it, "MB": v[2] / 1e6 / n_it}
                        for k, v in memcpy.items()},
    "sync_api_calls_per_iter": sync_calls / n_it,
  }
  print(json.dumps(out, indent=2))
  if a.json:
    with open(a.json, "w") as f:
      json.dump(out, f, indent=2)


if __name__ == "__main__":
  main()
