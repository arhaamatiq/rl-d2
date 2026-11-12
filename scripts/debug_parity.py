import sys
import torch
sys.path.insert(0, "tests")
from test_graph_env_parity import _det_cfg  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from babysteps.systems.graph_env import GraphedEnv, capture  # noqa: E402

n = 16
ref = ManagerBasedRlEnv(cfg=_det_cfg(n), device="cuda:0")
ge = GraphedEnv(_det_cfg(n), n)
act = torch.zeros(n, ge.num_actions, device="cuda:0")
ge.reset_all(); ge.step_body(act)
obs_r, _ = ref.reset(); obs_g = ge.reset_all()
print("t=-1 actor diff", float((obs_r["actor"] - obs_g["actor"][:n]).abs().max()))
eager = "--eager" in sys.argv
if not eager:
  g, out = capture(lambda: ge.step_body(act), warmup=0)
gen = torch.Generator(device="cuda:0").manual_seed(1)
om = ref.observation_manager
names = om.active_terms["actor"]; dims = [int(torch.tensor(d).prod()) for d in om.group_obs_term_dim["actor"]]
rnames = ref.reward_manager._term_names
for t in range(300):
  a = 0.8 * torch.randn(n, ge.num_actions, device="cuda:0", generator=gen)
  act.copy_(a)
  obs_r, rew_r, term_r, tout_r, _ = ref.step(a)
  if eager:
    out = ge.step_body(act)
  else:
    g.replay()
  obs_g, rew_g, term_g, tout_g = out
  d = (obs_r["actor"] - obs_g["actor"][:n]).abs()
  rd = (rew_r - rew_g[:n]).abs()
  if float(d.max()) > 1e-4 or float(rd.max()) > 1e-4:
    off = 0
    for nm, dm in zip(names, dims):
      print(f"  {nm:18s} maxdiff {float(d[:, off:off+dm].max()):.5f}"); off += dm
    sr = ref.reward_manager._step_reward; sg = ge.env.reward_manager._step_reward[:n]
    for i, nm in enumerate(rnames):
      print(f"  rew {nm:24s} {float((sr[:, i]-sg[:, i]).abs().max()):.5f}")
    env_bad = int(d.max(1).values.argmax())
    print("t", t, "first divergence; worst env", env_bad, "term", term_r.tolist(), "ep_len ref", ref.episode_length_buf.tolist(), "g", ge.env.episode_length_buf[:n].tolist())
    break
else:
  print("no divergence in 300 steps")
