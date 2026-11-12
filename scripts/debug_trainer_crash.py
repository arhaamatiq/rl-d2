import sys
import time

import torch

from babysteps.bar import load_bar
from babysteps.systems.graph_env import capture
from babysteps.train.fast_runner import FastTrainer
from babysteps.train.harness import RunClock
from babysteps.train.launch import apply_overrides
from babysteps.train.recipes import get_recipe

n, mode = int(sys.argv[1]), sys.argv[2]
r = get_recipe("graph_stock")
apply_overrides(r, [f"env.scene.num_envs={n}"])
tr = FastTrainer(r, None, RunClock(time.time()), load_bar())
if mode == "rollout_only":
  g, _ = capture(tr._rollout_body, warmup=1)
  print("captured", flush=True); g.replay(); torch.cuda.synchronize(); print("REPLAY OK", flush=True)
elif mode == "update_only":
  tr._rollout_body()
  g, _ = capture(tr.upd.body, warmup=1)
  print("captured", flush=True); g.replay(); torch.cuda.synchronize(); print("REPLAY OK", flush=True)
elif mode in ("both", "both_empty"):
  tr.build_graphs(); print("captured", flush=True)
  free, total = torch.cuda.mem_get_info()
  print(f"free {free/2**30:.2f} GiB of {total/2**30:.2f}; torch reserved {torch.cuda.memory_reserved()/2**30:.2f} GiB", flush=True)
  if mode == "both_empty":
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(); print(f"after empty_cache free {free/2**30:.2f} GiB", flush=True)
  tr.rollout_graph.replay(); torch.cuda.synchronize(); print("rollout replay OK", flush=True)
  tr.update_graph.replay(); torch.cuda.synchronize(); print("update replay OK", flush=True)
elif mode == "r_then_eager_update":
  g, _ = capture(tr._rollout_body, warmup=1); print("captured", flush=True)
  tr.upd.body(); torch.cuda.synchronize(); print("eager update OK", flush=True)
  g.replay(); torch.cuda.synchronize(); print("rollout replay OK", flush=True)
elif mode == "r_then_u_norestore":
  g, _ = capture(tr._rollout_body, warmup=1)
  gu, _ = capture(tr.upd.body, warmup=1, warp=False); print("captured", flush=True)
  g.replay(); torch.cuda.synchronize(); print("rollout replay OK", flush=True)
elif mode == "u_then_r":
  tr._rollout_body()
  gu, _ = capture(tr.upd.body, warmup=1, warp=False)
  g, _ = capture(tr._rollout_body, warmup=1); print("captured", flush=True)
  g.replay(); torch.cuda.synchronize(); print("rollout replay OK", flush=True)
  gu.replay(); torch.cuda.synchronize(); print("update replay OK", flush=True)
elif mode == "r_twice":
  g, _ = capture(tr._rollout_body, warmup=1); print("captured", flush=True)
  g.replay(); torch.cuda.synchronize(); print("rollout replay1 OK", flush=True)
  g.replay(); torch.cuda.synchronize(); print("rollout replay2 OK", flush=True)
elif mode == "r_freeze_then_u":
  import gc
  g, _ = capture(tr._rollout_body, warmup=1); gc.freeze()
  gu, _ = capture(tr.upd.body, warmup=1, warp=False); print("captured", flush=True)
  gc.collect()
  g.replay(); torch.cuda.synchronize(); print("rollout replay OK", flush=True)
  gu.replay(); torch.cuda.synchronize(); print("update replay OK", flush=True)
  for _ in range(20):
    g.replay(); gu.replay()
  torch.cuda.synchronize(); print("20 iterations OK", flush=True)
elif mode == "u_then_r_long":
  tr._rollout_body()
  gu, _ = capture(tr.upd.body, warmup=1, warp=False)
  g, _ = capture(tr._rollout_body, warmup=1); print("captured", flush=True)
  for _ in range(30):
    g.replay(); gu.replay()
  torch.cuda.synchronize(); print("30 iterations OK", flush=True)
  g2, _ = capture(tr._rollout_body, warmup=0); print("recaptured rollout", flush=True)
  for _ in range(30):
    g2.replay(); gu.replay()
  torch.cuda.synchronize(); print("30 more iterations after recapture OK", flush=True)
