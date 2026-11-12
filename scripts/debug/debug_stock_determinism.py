import sys

import torch

sys.path.insert(0, "tests")
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from test_graph_env_parity import _det_cfg  # noqa: E402

n = 16
for n2 in (16, 17):
  a_env = ManagerBasedRlEnv(cfg=_det_cfg(n), device="cuda:0")
  b_env = ManagerBasedRlEnv(cfg=_det_cfg(n2), device="cuda:0")
  oa, _ = a_env.reset(); ob, _ = b_env.reset()
  gen = torch.Generator(device="cuda:0").manual_seed(1)
  first = None; maxd = 0.0
  for t in range(300):
    act = 0.8 * torch.randn(n, a_env.action_manager.total_action_dim, device="cuda:0", generator=gen)
    actb = torch.cat([act, torch.zeros(n2 - n, act.shape[1], device="cuda:0")]) if n2 > n else act
    oa, ra, ta, _, _ = a_env.step(act); ob, rb, tb, _, _ = b_env.step(actb)
    d = float((oa["actor"] - ob["actor"][:n]).abs().max()); maxd = max(maxd, d)
    if first is None and d > 1e-4: first = t
  print(f"stock N={n} vs stock N={n2}: first obs divergence >1e-4 at step {first}, max diff {maxd:.3f}")
  del a_env, b_env
