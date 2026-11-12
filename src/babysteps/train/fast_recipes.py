"""Track A recipes (PPO family, same task). Each one is a row in the ablation table."""

from __future__ import annotations

from babysteps.train.recipes import Recipe, _stock_cfgs, register


@register
def graph_stock() -> Recipe:
  """Systems only: stock env count and PPO hyperparameters, rollout + update as CUDA graphs."""
  env_cfg, agent_cfg = _stock_cfgs()
  return Recipe("graph_stock", env_cfg, agent_cfg, runner="fast", options={"fused_adam": True})


def _finalize(recipe: Recipe) -> None:
  """Option-driven training-side edits (never touches the evaluation)."""
  import math

  from mjlab.managers.reward_manager import RewardTermCfg
  from mjlab.tasks.velocity import mdp

  o = recipe.options
  env = recipe.env_cfg
  std = o.get("fine_track_std")
  if std:
    # Extra, tighter tracking kernels on top of the stock ones: the stock yaw kernel
    # (std 0.71 rad/s) is nearly flat at the error levels the bar asks for.
    w = float(o.get("fine_track_w", 1.0))
    env.rewards["track_lin_fine"] = RewardTermCfg(
      func=mdp.track_linear_velocity, weight=w, params={"command_name": "twist", "std": float(std)})
    env.rewards["track_ang_fine"] = RewardTermCfg(
      func=mdp.track_angular_velocity, weight=w, params={"command_name": "twist", "std": float(std)})
  assert math.isfinite(float(o.get("fine_track_w", 1.0)))
  ang_std = o.get("fine_ang_std")
  if ang_std:
    # Yaw-only fine kernel (the both-axes version traded yaw for linear precision).
    env.rewards["track_ang_fine"] = RewardTermCfg(
      func=mdp.track_angular_velocity, weight=float(o.get("fine_ang_w", 1.0)),
      params={"command_name": "twist", "std": float(ang_std)})
  if o.get("loose_yaw_posture"):
    # Stock posture reward holds hip/waist yaw to std 0.15/0.2 rad while moving, which
    # fights turning in place; loosen only those joints in the walking regime.
    sw = dict(env.rewards["pose"].params["std_walking"])
    sw[r".*hip_yaw.*"] = float(o.get("loose_yaw_posture"))
    sw[r".*waist_yaw.*"] = float(o.get("loose_yaw_posture"))
    env.rewards["pose"].params["std_walking"] = sw


@register
def track_a_dev() -> Recipe:
  """Development recipe: graphed systems + options set with --set opt.<name>=..."""
  env_cfg, agent_cfg = _stock_cfgs()
  return Recipe("track_a_dev", env_cfg, agent_cfg, runner="fast", options={"fused_adam": True},
                finalize=_finalize)


# ---- named recipes: one per ablation row (cumulative), and the record ----------------


def _graphed(name: str, *, kl=0.01, symmetry=False, epochs=5, amp=None, yaw_w=None, fine_ang=None,
             lin_w=None) -> Recipe:
  env_cfg, agent_cfg = _stock_cfgs()
  agent_cfg.algorithm.desired_kl = kl
  agent_cfg.algorithm.num_learning_epochs = epochs
  opts = {"fused_adam": True, "symmetry": symmetry, "amp": amp}
  if fine_ang:
    opts["fine_ang_std"] = fine_ang
  if yaw_w:
    env_cfg.rewards["track_angular_velocity"].weight = yaw_w
  if lin_w:
    env_cfg.rewards["track_linear_velocity"].weight = lin_w
  return Recipe(name, env_cfg, agent_cfg, runner="fast", options=opts, finalize=_finalize)


@register
def abl_kl02() -> Recipe:
  return _graphed("abl_kl02", kl=0.02)


@register
def abl_kl02_sym() -> Recipe:
  return _graphed("abl_kl02_sym", kl=0.02, symmetry=True)


@register
def abl_kl02_sym_e8() -> Recipe:
  return _graphed("abl_kl02_sym_e8", kl=0.02, symmetry=True, epochs=8)


@register
def abl_kl02_sym_e8_bf16() -> Recipe:
  return _graphed("abl_kl02_sym_e8_bf16", kl=0.02, symmetry=True, epochs=8, amp="bf16")


@register
def track_a_fine() -> Recipe:
  return _graphed("track_a_fine", kl=0.02, symmetry=True, epochs=8, amp="bf16", fine_ang=0.25)


@register
def track_a_yaw4() -> Recipe:
  return _graphed("track_a_yaw4", kl=0.02, symmetry=True, epochs=8, amp="bf16", yaw_w=4.0)


@register
def track_a() -> Recipe:
  """The Track A record recipe (chosen by 3-seed validation on dev seeds 201-203)."""
  return _graphed("track_a", kl=0.02, symmetry=True, epochs=8, amp="bf16", yaw_w=4.0)


@register
def cand_lin3_yaw4() -> Recipe:
  return _graphed("cand_lin3_yaw4", kl=0.02, symmetry=True, epochs=8, amp="bf16", yaw_w=4.0, lin_w=3.0)


@register
def cand_lin2_yaw3() -> Recipe:
  return _graphed("cand_lin2_yaw3", kl=0.02, symmetry=True, epochs=8, amp="bf16", yaw_w=3.0)


@register
def cand_lin3_yaw3() -> Recipe:
  return _graphed("cand_lin3_yaw3", kl=0.02, symmetry=True, epochs=8, amp="bf16", yaw_w=3.0, lin_w=3.0)
