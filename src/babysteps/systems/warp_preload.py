"""Compile Warp kernel modules in parallel instead of lazily, one by one.

With cold caches (the benchmark wipes them), MuJoCo Warp's ~50 kernel modules are
compiled the first time each kernel launches, serially: ~42 s of a stock cold start on
the reference host. ``warp.force_load`` compiles every registered module with a thread
pool. Called once after imports and once after env construction (some modules are
created while the model is set up). Anything registered later still compiles lazily.
"""

from __future__ import annotations

import os
import time


def set_parallel_loading(workers: int | None = None) -> int:
  """Raise Warp's module-loading thread count (default min(cpu, 4)). mjlab's graph
  capture calls ``force_load`` with the config default, so this alone parallelizes
  the bulk of the cold-start compilation."""
  import warp as wp

  n = workers or max(1, (os.cpu_count() or 4))
  wp.config.load_module_max_workers = n
  return n


def preload(device: str = "cuda:0", workers: int | None = None) -> dict:
  import warp as wp

  t0 = time.time()
  wp.init()
  n = workers or max(1, (os.cpu_count() or 4))
  wp.force_load(device=device, max_workers=n)
  return {"seconds": time.time() - t0, "workers": n}
