"""Training recipes.

A recipe returns the env config, the agent config and the runner class. ``stock``
is mjlab's shipped G1 flat-velocity PPO setup with the README's 4096 envs; every
other recipe is a Track A entry built on the same task (same rewards,
observations, terminations, robot, physics) and must stay inside the PPO family.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

TASK_ID = "Mjlab-Velocity-Flat-Unitree-G1"


@dataclass
class Recipe:
  name: str
  env_cfg: object
  agent_cfg: object
  runner: str  # "stock" | "fast"
  options: dict = field(default_factory=dict)
  finalize: Callable[["Recipe"], None] | None = None  # applies option-driven cfg edits


RECIPES: dict[str, Callable[[], Recipe]] = {}


def register(fn: Callable[[], Recipe]) -> Callable[[], Recipe]:
  RECIPES[fn.__name__] = fn
  return fn


def _stock_cfgs():
  import mjlab.tasks  # noqa: F401  (populates the registry)
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  env_cfg = load_env_cfg(TASK_ID)
  agent_cfg = load_rl_cfg(TASK_ID)
  env_cfg.scene.num_envs = 4096
  agent_cfg.logger = "tensorboard"
  return env_cfg, agent_cfg


@register
def stock() -> Recipe:
  env_cfg, agent_cfg = _stock_cfgs()
  return Recipe("stock", env_cfg, agent_cfg, runner="stock")


def get_recipe(name: str) -> Recipe:
  if name not in RECIPES:
    # Track A recipes live in their own module to keep this one stable.
    from babysteps.train import fast_recipes  # noqa: F401
  if name not in RECIPES:
    raise KeyError(f"unknown recipe {name!r}; known: {sorted(RECIPES)}")
  return RECIPES[name]()
