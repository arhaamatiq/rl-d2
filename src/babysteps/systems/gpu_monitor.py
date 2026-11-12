"""Background NVML sampler for GPU utilization during training.

NVML ``utilization.gpu`` is the percentage of sample periods in which at least
one kernel was executing. It says nothing about how full the SMs were, so the
Nsight-derived "kernel coverage" (union of kernel intervals / wall time) is
reported next to it in the profile write-up.
"""

from __future__ import annotations

import threading
import time


class GpuMonitor:
  def __init__(self, device_index: int = 0, period_s: float = 0.1):
    self.period = period_s
    self.samples: list[tuple[float, int, int, float]] = []
    self._stop = threading.Event()
    self._ok = False
    try:
      import pynvml

      pynvml.nvmlInit()
      self._nvml = pynvml
      self._h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
      self._ok = True
    except Exception:
      self._nvml = None
    self._th = threading.Thread(target=self._run, daemon=True)

  def start(self) -> "GpuMonitor":
    if self._ok:
      self._th.start()
    return self

  def _run(self) -> None:
    n = self._nvml
    while not self._stop.is_set():
      try:
        u = n.nvmlDeviceGetUtilizationRates(self._h)
        p = n.nvmlDeviceGetPowerUsage(self._h) / 1000.0
        self.samples.append((time.time(), int(u.gpu), int(u.memory), float(p)))
      except Exception:
        pass
      time.sleep(self.period)

  def stop(self) -> None:
    self._stop.set()
    if self._th.is_alive():
      self._th.join(timeout=2.0)

  def summary(self, t_from: float | None = None) -> dict:
    s = [x for x in self.samples if t_from is None or x[0] >= t_from]
    if not s:
      return {"gpu_util_mean": None, "n": 0}
    g = [x[1] for x in s]
    p = [x[3] for x in s]
    return {
      "gpu_util_mean": sum(g) / len(g),
      "gpu_power_w_mean": sum(p) / len(p),
      "n": len(s),
    }
