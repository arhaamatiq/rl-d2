"""Track A runner: rsl_rl PPO with the rollout and the update each replayed as one
CUDA graph.

Per iteration the host does: replay(rollout graph) -> replay(update graph) ->
read a 9-float stats vector (one sync) -> maybe checkpoint. Inside the rollout
graph, for each of the T steps: actor/critic forward + sampling, the graphed env
step (physics + mjlab managers), storage write, normalizer update; then GAE.
"""

from __future__ import annotations

import time
from dataclasses import asdict

import torch
from tensordict import TensorDict

from babysteps.systems import nvtx
from babysteps.systems.graph_env import GraphedEnv, capture
from babysteps.systems.graphed_ppo import (
  GraphedPPOUpdate,
  make_capturable_gaussian,
  make_inplace_normalizer,
)
from babysteps.train.harness import CheckpointWriter, IterLog


class _Shim:
  """Minimal VecEnv surface that rsl_rl's PPO.construct_algorithm reads."""

  def __init__(self, num_envs, num_actions, cfg):
    self.num_envs = num_envs
    self.num_actions = num_actions
    self.cfg = cfg


class FastTrainer:
  def __init__(self, recipe, run_dir, clock, bar, device="cuda:0"):
    from rsl_rl.algorithms import PPO

    opts = recipe.options
    self.opts = opts
    self.device = device
    self.clock = clock
    env_cfg, agent_cfg = recipe.env_cfg, recipe.agent_cfg
    self.N = env_cfg.scene.num_envs
    self.genv = GraphedEnv(env_cfg, self.N, device=device)
    clock.mark("env_built")
    env = self.genv.env
    self.T = agent_cfg.num_steps_per_env
    obs_full = self.genv.reset_all()
    self.obs = TensorDict({k: v[: self.N].clone() for k, v in obs_full.items()}, batch_size=[self.N])
    cfg = asdict(agent_cfg)
    for key in ("actor", "critic"):
      for opt in ("cnn_cfg", "distribution_cfg"):
        if cfg[key].get(opt) is None:
          cfg[key].pop(opt, None)
      if cfg[key].get("rnn_type") is None:
        for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
          cfg[key].pop(opt, None)
    cfg["multi_gpu"] = None
    self.cfg = cfg
    self.alg = PPO.construct_algorithm(self.obs, _Shim(self.N, self.genv.num_actions, env.cfg), cfg, device)
    for m in (self.alg.actor, self.alg.critic):
      make_inplace_normalizer(m.obs_normalizer)
    make_capturable_gaussian(self.alg.actor.distribution)
    self.alg.train_mode()
    sym = None
    if opts.get("symmetry"):
      from babysteps.systems import symmetry

      sym = symmetry.build(env, device)
    self.upd = GraphedPPOUpdate(
      self.alg,
      fused_adam=opts.get("fused_adam", True),
      amp_dtype={"bf16": torch.bfloat16, None: None}[opts.get("amp")],
      symmetry=sym,
    )
    # init_at_random_ep_len, as the stock runner does
    env.episode_length_buf[: self.N] = torch.randint_like(
      env.episode_length_buf[: self.N], high=int(env.max_episode_length)
    )
    # on-device episode statistics
    self.cur_ret = torch.zeros(self.N, device=device)
    self.cur_len = torch.zeros(self.N, device=device)
    self.ep_stats = torch.zeros(4, device=device)  # sum_ret, sum_len, n_done, n_nan
    self.rollout_graph = None
    self.update_graph = None
    self.stats_host = [torch.zeros(9, pin_memory=True) for _ in range(2)]
    self.stats_evt = [torch.cuda.Event() for _ in range(2)]

  # ---- graph bodies ------------------------------------------------------------
  def _rollout_body(self) -> None:
    alg, N = self.alg, self.N
    alg.storage.clear()
    self.ep_stats.zero_()
    with torch.inference_mode():
      obs = self.obs
      for _ in range(self.T):
        actions = alg.act(obs)
        o, rew, term, tout = self.genv.step_body(actions)
        obs = TensorDict({k: v[:N] for k, v in o.items()}, batch_size=[N])
        rew, term, tout = rew[:N], term[:N], tout[:N]
        dones = (term | tout).to(torch.long)
        alg.process_env_step(obs, rew, dones, {"time_outs": tout})
        d = dones.bool()
        self.cur_ret += rew
        self.cur_len += 1.0
        self.ep_stats[0] += (self.cur_ret * d).sum()
        self.ep_stats[1] += (self.cur_len * d).sum()
        self.ep_stats[2] += d.sum()
        self.ep_stats[3] += torch.isnan(rew).sum()
        self.cur_ret.mul_(~d)
        self.cur_len.mul_(~d)
      alg.compute_returns(obs)
      self.upd.adv.copy_(alg.storage.advantages)
      for k in self.obs.keys():
        self.obs[k].copy_(obs[k])

  def build_graphs(self) -> None:
    snap = self.upd.snapshot()
    norm_snap = [
      {k: v.clone() for k, v in m.obs_normalizer.state_dict().items()} for m in (self.alg.actor, self.alg.critic)
    ]
    # Capture order matters. Capturing any graph *after* the rollout graph (which
    # contains MuJoCo Warp's conditional solver nodes) made the rollout graph's
    # next launch segfault inside cuGraphLaunch at >= 8k envs (driver 595.91,
    # CUDA 13.2). Update first, rollout last, has been stable at 8k and 16k envs
    # including later rollout re-captures.
    self._rollout_body()  # eager pass so the update warm-up sees real data
    self.update_graph, _ = capture(self.upd.body, warmup=1, warp=False)
    self.clock.mark("update_graph_captured")
    self.rollout_graph, _ = capture(self._rollout_body, warmup=1)
    self.clock.mark("rollout_graph_captured")
    # undo the warm-up's optimizer steps and normalizer updates
    self.upd.restore(snap)
    with torch.no_grad():
      for m, s in zip((self.alg.actor, self.alg.critic), norm_snap, strict=True):
        for k, v in m.obs_normalizer.state_dict().items():
          v.copy_(s[k])
    self.cur_ret.zero_()
    self.cur_len.zero_()

  # ---- main loop -----------------------------------------------------------------
  def learn(self, max_iterations, budget_s, ckpt: CheckpointWriter, iterlog: IterLog, prof=None):
    clock = self.clock
    steps_per_iter = self.T * self.N
    clock.t_first_iter = time.time()
    clock.mark("first_iter_start")
    it = 0
    for it in range(max_iterations):
      if prof is not None and it == prof[0]:
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
      nvtx.push(f"iter {it}")
      t0 = time.time()
      nvtx.push("rollout(graph)")
      self.rollout_graph.replay()
      nvtx.pop()
      nvtx.push("update(graph)")
      self.update_graph.replay()
      nvtx.pop()
      if self.genv.host_step_counters(self.T):
        nvtx.push("recapture")
        self.rollout_graph, _ = capture(self._rollout_body, warmup=0)
        nvtx.pop()
      # Stats readback without a sync: device->pinned-host copy + event, consumed
      # one iteration later, so the host can enqueue the next iteration's graphs
      # while this one is still running.
      slot = it % 2
      self.stats_host[slot].copy_(torch.cat([self.ep_stats, self.upd.stats]), non_blocking=True)
      self.stats_evt[slot].record()
      t1 = time.time()
      env_steps = (it + 1) * steps_per_iter
      if it > 0:
        prev = 1 - slot
        nvtx.push("wait.prev_stats")
        self.stats_evt[prev].synchronize()
        nvtx.pop()
        s = self.stats_host[prev].tolist()
        n_done = max(s[2], 1.0)
        iterlog.write({
          "it": it - 1, "t_wall_s": clock.wall(), "iter_s": t1 - t0, "env_steps": env_steps - steps_per_iter,
          "mean_return": s[0] / n_done, "mean_ep_len": s[1] / n_done, "episodes": s[2],
          "nan": s[3], "value_loss": s[4], "surrogate": s[5], "entropy": s[6], "kl": s[7], "lr": s[8],
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
      if clock.wall() >= budget_s:
        break
      if ckpt.stop_requested():
        break
    ckpt.write(self.alg.get_policy(), it, (it + 1) * steps_per_iter, {"final": True})


def run_fast(recipe, run_dir, clock, bar, args, prof):
  trainer = FastTrainer(recipe, run_dir, clock, bar, device=args.device)
  trainer.build_graphs()
  clock.mark("runner_built")
  ckpt = CheckpointWriter(run_dir, clock, bar.raw["checkpoint_interval_s"],
                          activation=recipe.agent_cfg.actor.activation)
  iterlog = IterLog(run_dir)
  trainer.learn(recipe.agent_cfg.max_iterations, args.budget_min * 60.0, ckpt, iterlog, prof)
  iterlog.close()
  return trainer
