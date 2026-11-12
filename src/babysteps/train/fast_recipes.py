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
