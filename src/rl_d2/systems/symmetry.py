"""Left-right mirror maps for the G1 actor observation and action vectors.

Mirroring about the sagittal plane: swap left/right joints, negate roll and yaw
joints, and flip the lateral / yaw components of base velocities, gravity and the
command. The default pose is mirror-symmetric (checked at build time), so joint
positions relative to the default mirror with the same map.
Used for symmetry data augmentation of the PPO actor loss (Mittal et al., ICRA 2024).
"""

from __future__ import annotations

import torch

_VEC_SIGNS = {
  "base_lin_vel": (1.0, -1.0, 1.0),
  "base_ang_vel": (-1.0, 1.0, -1.0),
  "projected_gravity": (1.0, -1.0, 1.0),
  "command": (1.0, -1.0, -1.0),
}


def _mirror_name(n: str) -> str:
  return n.replace("left_", "\0").replace("right_", "left_").replace("\0", "right_")


def joint_mirror(names: list[str]) -> tuple[list[int], list[float]]:
  perm = [names.index(_mirror_name(n)) for n in names]
  sign = [-1.0 if ("roll" in n or "yaw" in n) else 1.0 for n in names]
  return perm, sign


def build(env, device) -> dict:
  robot = env.scene["robot"]
  names = list(robot.joint_names)
  jperm, jsign = joint_mirror(names)
  q0 = robot.data.default_joint_pos[0]
  assert torch.allclose(q0[jperm] * torch.tensor(jsign, device=q0.device), q0, atol=1e-6), "default pose not symmetric"
  am = env.action_manager
  assert am.total_action_dim == len(names)  # action order == joint order for this task
  om = env.observation_manager
  perm, sign, off = [], [], 0
  for term, dim in zip(om.active_terms["actor"], om.group_obs_term_dim["actor"], strict=True):
    d = int(torch.tensor(dim).prod())
    if term in _VEC_SIGNS:
      perm += [off + i for i in range(d)]
      sign += list(_VEC_SIGNS[term])
    elif term in ("joint_pos", "joint_vel", "actions"):
      assert d == len(names)
      perm += [off + p for p in jperm]
      sign += jsign
    else:
      raise KeyError(f"no mirror rule for actor observation term {term!r}")
    off += d
  t = lambda x, dt: torch.tensor(x, dtype=dt, device=device)  # noqa: E731
  return {
    "obs_perm": t(perm, torch.long), "obs_sign": t(sign, torch.float32),
    "act_perm": t(jperm, torch.long), "act_sign": t(jsign, torch.float32),
  }
