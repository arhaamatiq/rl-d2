"""Find host<->device synchronization points in each stage of the stock env step.

Runs every stage of ManagerBasedRlEnv.step separately under
torch.cuda.set_sync_debug_mode("error") and reports the first syncing call site.
"""

from __future__ import annotations

import traceback

import torch


def main():
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
  cfg.scene.num_envs = 256
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  env.reset()
  a = 0.2 * torch.randn(256, env.action_manager.total_action_dim, device="cuda:0")
  for _ in range(5):
    env.step(a)
  dt = env.step_dt
  stages = {
    "action.process": lambda: env.action_manager.process_action(a),
    "action.apply": lambda: env.action_manager.apply_action(),
    "scene.write": lambda: env.scene.write_data_to_sim(),
    "sim.step": lambda: env.sim.step(),
    "scene.update": lambda: env.scene.update(dt=env.physics_dt),
    "metrics.substep": lambda: env.metrics_manager.compute_substep(),
    "termination": lambda: env.termination_manager.compute(),
    "reward": lambda: env.reward_manager.compute(dt=dt),
    "metrics": lambda: env.metrics_manager.compute(),
    "event.interval": lambda: env.event_manager.apply(mode="interval", dt=dt),
    "sim.forward": lambda: env.sim.forward(),
    "command": lambda: env.command_manager.compute(dt=dt),
    "sim.sense": lambda: env.sim.sense(),
    "obs": lambda: env.observation_manager.compute(update_history=True),
  }
  for name, fn in stages.items():
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
      fn()
      res = "sync-free"
    except RuntimeError as e:
      tb = traceback.extract_tb(e.__traceback__)
      site = [f"{f.filename.split('site-packages/')[-1]}:{f.lineno} {f.line}" for f in tb[-4:]]
      res = "SYNC at " + " <- ".join(reversed(site))
    finally:
      torch.cuda.set_sync_debug_mode("default")
    print(f"{name:16s} {res}")


if __name__ == "__main__":
  main()
