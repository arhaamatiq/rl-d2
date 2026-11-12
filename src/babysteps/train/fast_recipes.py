"""Track A recipes (PPO family, same task). Each one is a row in the ablation table."""

from __future__ import annotations

from babysteps.train.recipes import Recipe, _stock_cfgs, register


@register
def graph_stock() -> Recipe:
  """Systems only: stock env count and PPO hyperparameters, rollout + update as CUDA graphs."""
  env_cfg, agent_cfg = _stock_cfgs()
  return Recipe("graph_stock", env_cfg, agent_cfg, runner="fast", options={"fused_adam": True})
