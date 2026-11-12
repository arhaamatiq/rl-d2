import torch
import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
from rl_d2.systems.graph_env import GraphedEnv, capture

ge = GraphedEnv(load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1"), 4096)
ge.reset_all()
env = ge.env
def t(fn, n=50):
  g, _ = capture(fn, warmup=1)
  for _ in range(3): g.replay()
  torch.cuda.synchronize()
  s, e = torch.cuda.Event(True), torch.cuda.Event(True)
  s.record()
  for _ in range(n): g.replay()
  e.record(); torch.cuda.synchronize()
  return s.elapsed_time(e) / n
print("forward ms", round(t(lambda: env.sim.forward()), 3))
print("sense ms", round(t(lambda: env.sim.sense()), 3))
print("scene.update ms", round(t(lambda: env.scene.update(dt=env.physics_dt)), 3))
