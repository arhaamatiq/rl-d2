"""Sync-free, CUDA-graph-capturable step for mjlab manager-based envs.

Profiling the stock loop showed the GPU idle ~70% of the time: every env step runs
~1,100 small kernels from Python and hits ~95 host<->device synchronizations, while
the physics itself is already a CUDA graph. This module makes the *whole* env step
(physics + mjlab's own action/termination/reward/event/command/observation code)
capturable into one CUDA graph without rewriting the task:

* **Dummy world.** The env is built with ``N + 1`` worlds. Reset / resample /
  push indices are produced with ``torch.nonzero_static(mask[:N], size=N,
  fill_value=N)``: a fixed-size index tensor padded with the dummy world's index.
  mjlab's own ``reset(env_ids)`` code then runs unchanged with static shapes; the
  padded entries just write into the dummy world, which the learner never sees.
* **Raw kernels inside the capture.** mjlab's per-call Warp graphs are disabled
  and MuJoCo Warp kernels are recorded directly into the PyTorch graph through
  Warp's external-capture API.
* **Three sync sites patched** (found with ``scripts/audit_syncs.py``): the
  command term's ``nonzero`` resample/heading logic (rewritten with masks, same
  distribution), the raycast yaw-alignment ``if singular.any()`` branch
  (rewritten with ``torch.where``, same math), and Python-list body/site indices
  in reward terms (converted to slices / device index tensors). Reset-path
  ``.item()`` calls used only for logging are dropped.

Everything else (rewards, observations, terminations, physics) is mjlab's code.
``tests/test_graph_env_parity.py`` steps a stock env and a graphed env in lockstep
under a deterministic config and checks observations, rewards and dones match.
"""

from __future__ import annotations

import math

import torch
import warp as wp

# ----------------------------------------------------------------------------------
# Patches (applied per instance where possible)
# ----------------------------------------------------------------------------------


def _ids_to_static(ids, device):
  if isinstance(ids, list) and len(ids) > 0 and all(isinstance(i, int) for i in ids):
    if ids == list(range(ids[0], ids[0] + len(ids))):
      return slice(ids[0], ids[0] + len(ids))
    return torch.tensor(ids, dtype=torch.long, device=device)
  return ids


def patch_entity_cfg_ids(env) -> list[str]:
  """Turn Python-list id fields of SceneEntityCfg params into slices / device tensors."""
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  changed = []
  for mgr_name in ("reward_manager", "observation_manager", "termination_manager", "event_manager"):
    mgr = getattr(env, mgr_name, None)
    if mgr is None:
      continue
    cfgs = []
    if hasattr(mgr, "_term_cfgs"):
      tc = mgr._term_cfgs
      if isinstance(tc, dict):
        for v in tc.values():
          cfgs.extend(v)
      else:
        cfgs.extend(tc)
    if hasattr(mgr, "_group_obs_term_cfgs"):
      for v in mgr._group_obs_term_cfgs.values():
        cfgs.extend(v)
    if hasattr(mgr, "_mode_term_cfgs"):
      for mode, v in mgr._mode_term_cfgs.items():
        if mode in ("interval", "reset", "step"):
          cfgs.extend(v)
    for tc in cfgs:
      for pv in (getattr(tc, "params", None) or {}).values():
        if isinstance(pv, SceneEntityCfg):
          for attr in ("body_ids", "joint_ids", "site_ids", "geom_ids"):
            old = getattr(pv, attr, None)
            new = _ids_to_static(old, env.device)
            if new is not old:
              setattr(pv, attr, new)
              changed.append(f"{mgr_name}:{attr}")
  return changed


def _extract_yaw_rotation_nosync(self, rot_mat: torch.Tensor) -> torch.Tensor:
  """mjlab RayCastSensor._extract_yaw_rotation without the data-dependent branch."""
  batch_size = rot_mat.shape[0]
  x_proj = rot_mat[:, :, 0].clone()
  x_proj[:, 2] = 0
  x_norm = x_proj.norm(dim=1)
  singular = x_norm < 0.1
  y_proj = rot_mat[:, :, 1].clone()
  y_proj[:, 2] = 0
  y_proj = y_proj / y_proj.norm(dim=1).clamp(min=1e-6).unsqueeze(-1)
  x_from_y = torch.zeros_like(y_proj)
  x_from_y[:, 0] = y_proj[:, 1]
  x_from_y[:, 1] = -y_proj[:, 0]
  x_proj = torch.where(singular.unsqueeze(-1), x_from_y, x_proj)
  x_norm = torch.where(singular, torch.ones_like(x_norm), x_norm)
  x_norm = x_norm.clamp(min=1e-6)
  x_proj = x_proj / x_norm.unsqueeze(-1)
  yaw_mat = torch.zeros((batch_size, 3, 3), device=rot_mat.device, dtype=rot_mat.dtype)
  yaw_mat[:, 0, 0] = x_proj[:, 0]
  yaw_mat[:, 1, 0] = x_proj[:, 1]
  yaw_mat[:, 0, 1] = -x_proj[:, 1]
  yaw_mat[:, 1, 1] = x_proj[:, 0]
  yaw_mat[:, 2, 2] = 1
  return yaw_mat


def patch_raycast(env) -> int:
  n = 0
  sensors = getattr(env.scene, "sensors", None) or getattr(env.scene, "_sensors", {})
  for sensor in sensors.values():
    if hasattr(sensor, "_extract_yaw_rotation"):
      sensor._extract_yaw_rotation = _extract_yaw_rotation_nosync.__get__(sensor)
      n += 1
  return n


def patch_velocity_command(term, n_real: int, turn_frac: float = 0.0) -> None:
  """Static-shape, sync-free versions of UniformVelocityCommand's per-step logic.

  Same sampling distribution and update rules as mjlab 1.6.0; only the index
  handling changes (masked updates instead of ``nonzero`` / boolean indexing).
  """
  from mjlab.utils.lab_api.math import wrap_to_pi

  cfg = term.cfg
  dummy = n_real
  assert getattr(cfg, "init_velocity_prob", 0.0) == 0.0
  orig_compute = term.compute
  orig_update = term._update_command

  def _resample_command(env_ids):
    r = torch.empty(len(env_ids), device=term.device)
    term.vel_command_b[env_ids, 0] = r.uniform_(*cfg.ranges.lin_vel_x)
    term.vel_command_b[env_ids, 1] = r.uniform_(*cfg.ranges.lin_vel_y)
    term.vel_command_b[env_ids, 2] = r.uniform_(*cfg.ranges.ang_vel_z)
    if cfg.heading_command:
      term.heading_target[env_ids] = r.uniform_(*cfg.ranges.heading)
      term.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= cfg.rel_heading_envs
    term.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= cfg.rel_standing_envs
    term.is_world_env[env_ids] = r.uniform_(0.0, 1.0) <= cfg.rel_world_envs
    term.vel_command_w[env_ids] = term.vel_command_b[env_ids]
    term.is_forward_env[env_ids] = r.uniform_(0.0, 1.0) <= cfg.rel_forward_envs
    fwd = term.is_forward_env[env_ids]
    v = term.vel_command_b[env_ids]
    v0 = torch.where(fwd, v[:, 0].abs().clamp(min=0.3), v[:, 0])
    v1 = torch.where(fwd, torch.zeros_like(v[:, 1]), v[:, 1])
    v2 = torch.where(fwd, torch.zeros_like(v[:, 2]), v[:, 2])
    term.vel_command_b[env_ids] = torch.stack([v0, v1, v2], dim=1)
    if turn_frac > 0.0:
      # Training curriculum option: a share of commands become pure turn-in-place,
      # |wz| ~ U(0.25, max), no linear velocity, no heading control.
      turn = torch.rand(len(env_ids), device=term.device) < turn_frac
      lo, hi = 0.25, float(cfg.ranges.ang_vel_z[1])
      mag = torch.rand(len(env_ids), device=term.device) * (hi - lo) + lo
      sgn = torch.where(torch.rand(len(env_ids), device=term.device) < 0.5, -1.0, 1.0)
      v = term.vel_command_b[env_ids]
      tw = torch.stack([torch.zeros_like(mag), torch.zeros_like(mag), sgn * mag], dim=1)
      term.vel_command_b[env_ids] = torch.where(turn.unsqueeze(1), tw, v)
      term.vel_command_w[env_ids] = term.vel_command_b[env_ids]
      for flag in (term.is_heading_env, term.is_standing_env, term.is_world_env):
        flag[env_ids] = flag[env_ids] & ~turn

  def _update_command(env_ids=None):
    if env_ids is not None:  # eager reset path (initial env.reset), not captured
      return orig_update(env_ids)
    if cfg.heading_command:
      term.heading_error = wrap_to_pi(term.heading_target - term.robot.data.heading_w)
      wz = torch.clip(
        cfg.heading_control_stiffness * term.heading_error,
        min=cfg.ranges.ang_vel_z[0],
        max=cfg.ranges.ang_vel_z[1],
      )
      term.vel_command_b[:, 2] = torch.where(term.is_heading_env, wz, term.vel_command_b[:, 2])
    if cfg.rel_world_envs > 0.0:
      heading = term.robot.data.heading_w
      c, s = torch.cos(heading), torch.sin(heading)
      vx_w, vy_w = term.vel_command_w[:, 0], term.vel_command_w[:, 1]
      w = term.is_world_env
      term.vel_command_b[:, 0] = torch.where(w, c * vx_w + s * vy_w, term.vel_command_b[:, 0])
      term.vel_command_b[:, 1] = torch.where(w, -s * vx_w + c * vy_w, term.vel_command_b[:, 1])
    st = term.is_standing_env.unsqueeze(-1)
    term.vel_command_b.copy_(torch.where(st, torch.zeros_like(term.vel_command_b), term.vel_command_b))
    term.vel_command_w.copy_(torch.where(st, torch.zeros_like(term.vel_command_w), term.vel_command_w))

  def compute(dt, env_ids=None):
    if env_ids is not None:  # eager reset path (initial env.reset), not captured
      return orig_compute(dt, env_ids)
    term._update_metrics()
    term.time_left -= dt
    ids = torch.nonzero_static(term.time_left[:dummy] <= 0.0, size=dummy, fill_value=dummy).flatten()
    term.time_left[ids] = term.time_left[ids].uniform_(*cfg.resampling_time_range)
    _resample_command(ids)
    term.command_counter[ids] += 1
    _update_command(None)

  def reset(env_ids):
    for metric_value in term.metrics.values():
      metric_value[env_ids] = 0.0
    term.command_counter[env_ids] = 0
    term.time_left[env_ids] = term.time_left[env_ids].uniform_(*cfg.resampling_time_range)
    _resample_command(env_ids)
    term.command_counter[env_ids] += 1
    return {}

  term._resample_command = _resample_command
  term._update_command = _update_command
  term.compute = compute
  term.reset = reset


def patch_torcharray_setitem() -> None:
  """mjlab's TorchArray.__setitem__ switches to Warp's device stream for every
  write. Inside a capture that stream is not part of the graph. In the graphed
  path torch and Warp share one stream, so writes stay on the current stream."""
  from mjlab.sim.sim_data import TorchArray

  if getattr(TorchArray, "_bs_patched", False):
    return

  def __setitem__(self, idx, value):
    self._tensor[idx] = value

  TorchArray.__setitem__ = __setitem__
  TorchArray._bs_patched = True


def patch_scalar_setitem() -> None:
  """``t[cuda_index] = 0.0`` stages the Python scalar through a pageable host
  tensor (torch 2.9), which cannot be captured. mjlab's reset code uses this
  pattern throughout; route such scalars through a device 0-d tensor instead."""
  if getattr(torch.Tensor, "_bs_scalar_patched", False):
    return
  orig = torch.Tensor.__setitem__

  def _has_tensor(idx) -> bool:
    if isinstance(idx, torch.Tensor):
      return True
    return isinstance(idx, tuple) and any(isinstance(i, torch.Tensor) for i in idx)

  def __setitem__(self, idx, value):
    if isinstance(value, (bool, int, float)) and self.is_cuda and _has_tensor(idx):
      value = torch.full((), value, dtype=self.dtype, device=self.device)
    return orig(self, idx, value)

  torch.Tensor.__setitem__ = __setitem__
  torch.Tensor._bs_scalar_patched = True


def patch_event_sampling() -> None:
  """mjlab's _sample_se3_range builds its range tensor from a Python list on every
  call (a pageable host->device copy, illegal during capture). Cache it per range."""
  import mjlab.envs.mdp.events as ev

  if getattr(ev, "_bs_patched", False):
    return
  cache: dict = {}
  sample_uniform = ev.sample_uniform

  def _sample_se3_range(range_dict, shape, device):
    range_dict = range_dict or {}
    key = (tuple((k, tuple(range_dict.get(k, (0.0, 0.0)))) for k in ev._SE3_KEYS), str(device))
    t = cache.get(key)
    if t is None:
      t = torch.tensor([range_dict.get(k, (0.0, 0.0)) for k in ev._SE3_KEYS], device=device)
      cache[key] = t
    return sample_uniform(t[:, 0], t[:, 1], shape, device=device)

  ev._sample_se3_range = _sample_se3_range
  ev._bs_patched = True


def patch_logging_resets(env) -> None:
  """Reset-path bookkeeping that only feeds logs and forces ``.item()`` syncs."""
  tm = env.termination_manager

  def term_reset(env_ids=None):
    for term_cfg in tm._class_term_cfgs:
      term_cfg.func.reset(env_ids=env_ids)
    return {}

  tm.reset = term_reset
  rm = env.reward_manager

  def rew_reset(env_ids=None):
    for key in rm._episode_sums.keys():
      rm._episode_sums[key][env_ids] = 0.0
    for term_cfg in rm._class_term_cfgs:
      term_cfg.func.reset(env_ids=env_ids)
    return {}

  rm.reset = rew_reset


# ----------------------------------------------------------------------------------
# Graphed env
# ----------------------------------------------------------------------------------


class GraphedEnv:
  def __init__(self, env_cfg, num_envs: int, device: str = "cuda:0", turn_frac: float = 0.0):
    from mjlab.envs import ManagerBasedRlEnv

    env_cfg.scene.num_envs = num_envs + 1
    self.env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    self.N = num_envs
    self.device = device
    env = self.env
    self.patched_ids = patch_entity_cfg_ids(env)
    self.patched_raycast = patch_raycast(env)
    for term in env.command_manager._terms.values():
      patch_velocity_command(term, num_envs, turn_frac=turn_frac)
    patch_logging_resets(env)
    patch_torcharray_setitem()
    patch_event_sampling()
    patch_scalar_setitem()
    env.sim.use_cuda_graph = False  # raw Warp kernels; captured at the step level instead
    self.num_actions = env.action_manager.total_action_dim
    self.action_buf = torch.zeros(num_envs + 1, self.num_actions, device=device)
    self.command_dt = torch.full((num_envs + 1,), env.step_dt, device=device)
    self.step_dt = env.step_dt
    self.decimation = env.cfg.decimation

  # ---- env pieces -----------------------------------------------------------------
  def _interval_events(self) -> None:
    em = self.env.event_manager
    if "interval" not in em._mode_term_cfgs:
      return
    N = self.N
    for index, term_cfg in enumerate(em._mode_term_cfgs["interval"]):
      assert not term_cfg.is_global_time
      time_left = em._interval_term_time_left[index]
      time_left -= self.step_dt
      ids = torch.nonzero_static(time_left[:N] < 1e-6, size=N, fill_value=N).flatten()
      lower, upper = term_cfg.interval_range_s
      time_left[ids] = torch.rand(N, device=self.device) * (upper - lower) + lower
      term_cfg.func(self.env, ids, **term_cfg.params)
      level = getattr(term_cfg.func, "recompute", None)
      if level is not None and int(level) > 0:
        self.env.sim.recompute_constants(level)

  def _reset_idx(self, ids: torch.Tensor) -> None:
    env = self.env
    env.sim.reset(ids)
    env.scene.reset(ids)
    if "reset" in env.event_manager.available_modes:
      env.event_manager.apply(
        mode="reset", env_ids=ids, global_env_step_count=env._sim_step_counter // self.decimation
      )
    env.observation_manager.reset(ids)
    env.action_manager.reset(ids)
    env.reward_manager.reset(ids)
    env.metrics_manager.reset(ids)
    env.curriculum_manager.reset(ids)
    env.command_manager.reset(ids)
    env.event_manager.reset(ids)
    env.termination_manager.reset(ids)
    env.episode_length_buf[ids] = 0

  def step_body(self, action: torch.Tensor):
    """One control step for the N real envs. Graph-capturable (no host syncs)."""
    env, N = self.env, self.N
    self.action_buf[:N].copy_(action)
    env.action_manager.process_action(self.action_buf)
    for _ in range(self.decimation):
      env.action_manager.apply_action()
      env.scene.write_data_to_sim()
      env.sim.step()
      env.scene.update(dt=env.physics_dt)
      env.metrics_manager.compute_substep()
    env.episode_length_buf += 1
    reset_buf = env.termination_manager.compute()
    terminated = env.termination_manager.terminated
    time_outs = env.termination_manager.time_outs
    reward = env.reward_manager.compute(dt=self.step_dt)
    env.metrics_manager.compute()
    self._interval_events()
    ids = torch.nonzero_static(reset_buf[:N], size=N, fill_value=N).flatten()
    self._reset_idx(ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    self.command_dt.fill_(self.step_dt)
    self.command_dt[ids] = 0.0
    env.command_manager.compute(dt=self.command_dt)
    env.sim.sense()
    obs = env.observation_manager.compute(update_history=True)
    return obs, reward, terminated, time_outs

  def host_step_counters(self, n_steps: int) -> bool:
    """Advance host-side counters after ``n_steps`` graph steps; run the curriculum.
    Returns True if the curriculum changed constants baked into the graph."""
    env = self.env
    env.common_step_counter += n_steps
    env._sim_step_counter += n_steps * self.decimation
    before = self.curriculum_signature()
    env.curriculum_manager.compute(env_ids=None)
    return self.curriculum_signature() != before

  def curriculum_signature(self) -> tuple:
    sig = []
    for term in self.env.command_manager._terms.values():
      r = getattr(term.cfg, "ranges", None)
      if r is not None:
        sig.append((tuple(r.lin_vel_x), tuple(r.lin_vel_y), tuple(r.ang_vel_z)))
    return tuple(sig)

  def reset_all(self):
    obs, _ = self.env.reset()
    return obs


# ----------------------------------------------------------------------------------
# Capture helper: PyTorch graph with Warp kernels recorded via external capture
# ----------------------------------------------------------------------------------


class CapturedGraph:
  """A PyTorch CUDA graph plus the Warp graph object for the same capture.

  Warp's Graph owns resources the recorded nodes use (e.g. state for the
  conditional nodes of MuJoCo Warp's solver loop). If it is garbage-collected
  the PyTorch graph keeps pointers to freed resources and replay crashes. Keep
  both alive together."""

  def __init__(self, torch_graph, warp_graph):
    self.torch_graph = torch_graph
    self.warp_graph = warp_graph

  def replay(self) -> None:
    self.torch_graph.replay()


def capture(fn, warmup: int = 2, pool=None, warp: bool = True):
  """Capture ``fn()`` (torch + Warp work) into a CUDA graph. Returns (graph, outputs).
  ``warp=False`` for torch-only bodies (no Warp external-capture hooks)."""
  import gc

  s = torch.cuda.Stream()
  s.wait_stream(torch.cuda.current_stream())
  wps = wp.stream_from_torch(s)
  with torch.cuda.stream(s), wp.ScopedStream(wps, sync_enter=False):
    for _ in range(warmup):
      fn()
  torch.cuda.current_stream().wait_stream(s)
  torch.cuda.synchronize()
  g = torch.cuda.CUDAGraph()
  # A finalizer of a stale Warp object running mid-capture would be recorded
  # into this graph; keep the collector off while capturing.
  gc_was_enabled = gc.isenabled()
  gc.disable()
  try:
    wgraph = None
    with torch.cuda.graph(g, stream=s, pool=pool):
      if warp:
        wp.capture_begin(stream=wps, external=True)
        try:
          with wp.ScopedStream(wps, sync_enter=False):
            out = fn()
        finally:
          wgraph = wp.capture_end(stream=wps)
      else:
        out = fn()
  finally:
    if gc_was_enabled:
      gc.enable()
  torch.cuda.synchronize()
  return CapturedGraph(g, wgraph), out


def deg(x: float) -> float:
  return math.degrees(x)
