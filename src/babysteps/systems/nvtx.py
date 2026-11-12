"""NVTX helpers.

Ranges are pushed only when ``BS_NVTX=1`` so that timing runs pay nothing. When
enabled, the profile splits each iteration into:

  rollout.policy      actor/critic forward during data collection
  rollout.env         env.step (contains the ranges below)
    env.physics       MuJoCo Warp step (CUDA graph replay), x decimation
    env.managers      action/termination/reward/event/command/observation managers
    env.forward       mj_forward refresh before observations
  rollout.storage     writing the transition into the rollout buffer
  gae                 return / advantage computation
  update.forward      PPO minibatch forward (actor + critic)
  update.backward     loss.backward()
  update.optim        clip_grad_norm + optimizer.step()
  update.sync         host<->device syncs (.item(), adaptive-LR comparisons)
  ckpt                checkpoint write (device -> host copy)
"""

from __future__ import annotations

import contextlib
import functools
import os

ENABLED = os.environ.get("BS_NVTX", "0") == "1"

if ENABLED:
  import nvtx as _nvtx

  def push(name: str) -> None:
    _nvtx.push_range(name)

  def pop() -> None:
    _nvtx.pop_range()
else:

  def push(name: str) -> None:  # noqa: ARG001
    pass

  def pop() -> None:
    pass


@contextlib.contextmanager
def rng(name: str):
  push(name)
  try:
    yield
  finally:
    pop()


def wrap_method(obj, attr: str, name: str) -> None:
  """Replace ``obj.attr`` with a version wrapped in an NVTX range (no-op if disabled)."""
  if not ENABLED:
    return
  fn = getattr(obj, attr)

  @functools.wraps(fn)
  def wrapped(*a, **k):
    push(name)
    try:
      return fn(*a, **k)
    finally:
      pop()

  setattr(obj, attr, wrapped)


def instrument_env(env) -> None:
  """Attach NVTX ranges to the stages of an mjlab ManagerBasedRlEnv."""
  if not ENABLED:
    return
  u = env.unwrapped
  wrap_method(u.sim, "step", "env.physics")
  wrap_method(u.sim, "forward", "env.forward")
  wrap_method(u.sim, "sense", "env.sense")
  for mgr, name in (
    (u.action_manager, "apply_action"),
    (u.action_manager, "process_action"),
    (u.termination_manager, "compute"),
    (u.reward_manager, "compute"),
    (u.command_manager, "compute"),
    (u.observation_manager, "compute"),
    (u.metrics_manager, "compute"),
    (u.metrics_manager, "compute_substep"),
  ):
    wrap_method(mgr, name, f"env.managers.{type(mgr).__name__}.{name}")
  wrap_method(u.event_manager, "apply", "env.managers.EventManager.apply")
  wrap_method(u.scene, "update", "env.managers.scene.update")
  wrap_method(u.scene, "write_data_to_sim", "env.managers.scene.write")
  wrap_method(u, "_reset_idx", "env.reset_idx")
