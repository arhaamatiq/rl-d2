"""GPU time per phase of the graphed training iteration.

NVTX ranges cannot split work inside a CUDA graph, so each phase is captured as
its own graph and timed with CUDA events over repeated replays:
  policy inference (actor+critic forward, sampling, log-prob), env step
  (physics substeps / forward+sense / mjlab managers), rollout storage +
  normalizer update, GAE, PPO update forward+loss / backward / clip+optimizer.

  python scripts/bench_components.py [--set env.scene.num_envs=8192 ...] --json out.json
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from tensordict import TensorDict


def timed(g, reps=20) -> float:
  for _ in range(3):
    g.replay()
  torch.cuda.synchronize()
  s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
  s.record()
  for _ in range(reps):
    g.replay()
  e.record()
  torch.cuda.synchronize()
  return s.elapsed_time(e) / reps


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--recipe", default="graph_stock")
  ap.add_argument("--set", action="append", default=[])
  ap.add_argument("--json", default=None)
  a = ap.parse_args()

  from mjlab.utils.torch import configure_torch_backends

  from babysteps.bar import load_bar
  from babysteps.systems.graph_env import capture
  from babysteps.train.fast_runner import FastTrainer
  from babysteps.train.harness import RunClock
  from babysteps.train.launch import apply_overrides
  from babysteps.train.recipes import get_recipe

  configure_torch_backends()
  recipe = get_recipe(a.recipe)
  apply_overrides(recipe, a.set)
  tr = FastTrainer(recipe, None, RunClock(time.time()), load_bar())
  tr.build_graphs()
  alg, ge, N, T = tr.alg, tr.genv, tr.N, tr.T
  env = ge.env
  res = {"num_envs": N, "steps_per_env": T, "epochs": alg.num_learning_epochs,
         "minibatches": alg.num_mini_batches}
  res["rollout_graph_ms"] = timed(tr.rollout_graph, 5)
  res["update_graph_ms"] = timed(tr.update_graph, 5)

  obs = tr.obs
  act_buf = torch.zeros(N, ge.num_actions, device="cuda:0")

  def act():
    with torch.inference_mode():
      act_buf.copy_(alg.act(obs))

  g, _ = capture(act)
  res["policy_inference_ms_per_step"] = timed(g)

  g, _ = capture(lambda: ge.step_body(act_buf))
  res["env_step_ms_per_step"] = timed(g)

  def physics():
    for _ in range(ge.decimation):
      env.sim.step()

  g, _ = capture(physics)
  res["physics_ms_per_step"] = timed(g)

  def fwd_sense():
    env.sim.forward()
    env.sim.sense()

  g, _ = capture(fwd_sense)
  res["forward_sense_ms_per_step"] = timed(g)
  res["env_managers_ms_per_step"] = (
    res["env_step_ms_per_step"] - res["physics_ms_per_step"] - res["forward_sense_ms_per_step"]
  )

  rew = torch.zeros(N, device="cuda:0")
  dones = torch.zeros(N, dtype=torch.long, device="cuda:0")
  tout = torch.zeros(N, dtype=torch.bool, device="cuda:0")

  def act_store():
    with torch.inference_mode():
      alg.storage.clear()
      alg.act(obs)
      alg.process_env_step(TensorDict(dict(obs.items()), batch_size=[N]), rew, dones, {"time_outs": tout})

  g, _ = capture(act_store)
  res["storage_ms_per_step"] = timed(g) - res["policy_inference_ms_per_step"]

  def gae():
    with torch.inference_mode():
      alg.compute_returns(obs)

  g, _ = capture(gae)
  res["gae_ms"] = timed(g)

  snap = tr.upd.snapshot()
  for name, stages in (("fwd", ("fwd",)), ("fwd_bwd", ("fwd", "bwd")), ("full", ("fwd", "bwd", "opt"))):
    g, _ = capture(lambda st=stages: tr.upd.body(st))
    res[f"update_{name}_ms"] = timed(g, 5)
  tr.upd.restore(snap)
  res["update_forward_loss_ms"] = res["update_fwd_ms"]
  res["update_backprop_ms"] = res["update_fwd_bwd_ms"] - res["update_fwd_ms"]
  res["update_optimizer_ms"] = res["update_full_ms"] - res["update_fwd_bwd_ms"]
  res["per_iter_ms"] = {
    "simulation": T * (res["physics_ms_per_step"] + res["forward_sense_ms_per_step"]),
    "env_managers": T * res["env_managers_ms_per_step"],
    "policy_inference": T * res["policy_inference_ms_per_step"],
    "rollout_storage": T * res["storage_ms_per_step"],
    "gae": res["gae_ms"],
    "update_forward_loss": res["update_forward_loss_ms"],
    "backprop": res["update_backprop_ms"],
    "optimizer_step": res["update_optimizer_ms"],
  }
  print(json.dumps(res, indent=2))
  if a.json:
    with open(a.json, "w") as f:
      json.dump(res, f, indent=2)


if __name__ == "__main__":
  main()
