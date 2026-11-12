"""Stock runner under the harness.

``TimedVelocityRunner.learn`` is rsl_rl 5.4.2's ``OnPolicyRunner.learn`` line for
line, with four additions that every track shares:
  1. the first-iteration timestamp (for the "excluding startup" number),
  2. wall-clock checkpoints of the deterministic actor for the evaluator,
  3. a wall-clock budget after which the run stops,
  4. NVTX ranges (inactive unless BS_NVTX=1).
The PPO math, rollout order, logging and mjlab's own periodic saves are untouched.
"""

from __future__ import annotations

import os
import time

import torch
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from rsl_rl.utils import check_nan

from babysteps.systems import nvtx
from babysteps.train.harness import CheckpointWriter, IterLog, RunClock


def instrument_ppo(alg) -> None:
  """NVTX ranges inside PPO.update (profiling only)."""
  if not nvtx.ENABLED:
    return
  nvtx.wrap_method(alg, "update", "update")
  nvtx.wrap_method(alg, "compute_returns", "gae")
  nvtx.wrap_method(alg, "process_env_step", "rollout.storage")
  nvtx.wrap_method(alg, "act", "rollout.policy")
  nvtx.wrap_method(alg.optimizer, "step", "update.optim")
  orig_backward = torch.Tensor.backward

  def backward(self, *a, **k):
    nvtx.push("update.backward")
    try:
      return orig_backward(self, *a, **k)
    finally:
      nvtx.pop()

  torch.Tensor.backward = backward
  orig_clip = torch.nn.utils.clip_grad_norm_

  def clip(*a, **k):
    nvtx.push("update.clip_grad")
    try:
      return orig_clip(*a, **k)
    finally:
      nvtx.pop()

  torch.nn.utils.clip_grad_norm_ = clip
  orig_item = torch.Tensor.item

  def item(self):
    nvtx.push("sync.item")
    try:
      return orig_item(self)
    finally:
      nvtx.pop()

  torch.Tensor.item = item


class TimedVelocityRunner(VelocityOnPolicyRunner):
  harness_clock: RunClock
  harness_ckpt: CheckpointWriter
  harness_budget_s: float
  harness_iterlog: IterLog
  harness_profile_iters: tuple[int, int] | None = None

  def attach_harness(self, clock, ckpt, budget_s, iterlog, profile_iters=None):
    self.harness_clock = clock
    self.harness_ckpt = ckpt
    self.harness_budget_s = budget_s
    self.harness_iterlog = iterlog
    self.harness_profile_iters = profile_iters
    instrument_ppo(self.alg)

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    clock, ckpt = self.harness_clock, self.harness_ckpt
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )
    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()
    if self.is_distributed:
      self.alg.broadcast_parameters()
    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    steps_per_iter = self.cfg["num_steps_per_env"] * self.env.num_envs
    clock.t_first_iter = time.time()
    clock.mark("first_iter_start")
    prof = self.harness_profile_iters
    for it in range(start_it, total_it):
      if prof is not None and it == prof[0]:
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
      nvtx.push(f"iter {it}")
      start = time.time()
      nvtx.push("rollout")
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = self.alg.act(obs)
          nvtx.push("rollout.env")
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          nvtx.pop()
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, rewards, dones)
          obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
          self.alg.process_env_step(obs, rewards, dones, extras)
          intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
          self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
        stop = time.time()
        collect_time = stop - start
        start = stop
        self.alg.compute_returns(obs)
      nvtx.pop()
      loss_dict = self.alg.update()
      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it

      nvtx.push("log")
      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
      )
      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore
      nvtx.pop()

      # ---- harness ---------------------------------------------------------------
      env_steps = (it - start_it + 1) * steps_per_iter
      self.harness_iterlog.write({
        "it": it,
        "t_wall_s": clock.wall(),
        "collect_s": collect_time,
        "learn_s": learn_time,
        "env_steps": env_steps,
      })
      if ckpt.due():
        nvtx.push("ckpt")
        ckpt.write(self.alg.get_policy(), it, env_steps)
        nvtx.pop()
      nvtx.pop()
      if prof is not None and it == prof[1]:
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        break
      if clock.wall() >= self.harness_budget_s:
        break

    ckpt.write(self.alg.get_policy(), self.current_learning_iteration, env_steps, {"final": True})
    if self.logger.writer is not None:
      self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
      self.logger.stop_logging_writer()
