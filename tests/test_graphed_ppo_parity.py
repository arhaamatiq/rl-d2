"""GraphedPPOUpdate vs rsl_rl 5.4.2 PPO.update on identical storage and permutation."""

import pytest
import torch
from tensordict import TensorDict

pytestmark = pytest.mark.gpu


class _Env:
  num_envs = 256
  num_actions = 12
  cfg = None


def _make_alg(device):
  from rsl_rl.algorithms import PPO

  n, T = _Env.num_envs, 24
  obs = TensorDict({"actor": torch.randn(n, 40, device=device), "critic": torch.randn(n, 50, device=device)},
                   batch_size=[n])
  cfg = {
    "num_steps_per_env": T, "obs_groups": {"actor": ("actor",), "critic": ("critic",)}, "multi_gpu": None,
    "actor": {"class_name": "MLPModel", "hidden_dims": (64, 64), "activation": "elu", "obs_normalization": True,
              "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}},
    "critic": {"class_name": "MLPModel", "hidden_dims": (64, 64), "activation": "elu", "obs_normalization": True},
    "algorithm": {"class_name": "PPO", "value_loss_coef": 1.0, "use_clipped_value_loss": True, "clip_param": 0.2,
                  "entropy_coef": 0.01, "num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": 1e-3,
                  "schedule": "adaptive", "gamma": 0.99, "lam": 0.95, "desired_kl": 0.01, "max_grad_norm": 1.0,
                  "rnd_cfg": None, "symmetry_cfg": None},
  }
  alg = PPO.construct_algorithm(obs, _Env(), cfg, device)
  alg.train_mode()
  # fill storage with a plausible rollout
  st = alg.storage
  for _ in range(T):
    o = TensorDict({"actor": torch.randn(n, 40, device=device), "critic": torch.randn(n, 50, device=device)},
                   batch_size=[n])
    alg.act(o)
    alg.process_env_step(o, torch.randn(n, device=device), (torch.rand(n, device=device) < 0.02).long(), {})
  alg.compute_returns(o)
  return alg, st


def test_update_matches_rsl_rl():
  from babysteps.systems.graphed_ppo import GraphedPPOUpdate, make_inplace_normalizer

  device = "cuda:0"
  torch.manual_seed(0)
  alg_a, st = _make_alg(device)
  torch.manual_seed(0)
  alg_b, st_b = _make_alg(device)
  assert torch.equal(st.actions, st_b.actions)
  B = st.num_envs * st.num_transitions_per_env
  perm = torch.randperm(B, device=device)

  orig = torch.randperm
  torch.randperm = lambda *a, **k: perm.clone()
  try:
    alg_a.update()
  finally:
    torch.randperm = orig

  for m in (alg_b.actor, alg_b.critic):
    make_inplace_normalizer(m.obs_normalizer)
  upd = GraphedPPOUpdate(alg_b, fused_adam=False)
  upd.adv.copy_(alg_b.storage.advantages)
  upd.perm_override = perm
  upd.body()
  torch.cuda.synchronize()

  worst = 0.0
  for (na, pa), (nb, pb) in zip(
    list(alg_a.actor.named_parameters()) + list(alg_a.critic.named_parameters()),
    list(alg_b.actor.named_parameters()) + list(alg_b.critic.named_parameters()),
  ):
    assert na == nb
    worst = max(worst, float((pa - pb).abs().max()))
  print({"max_param_diff": worst, "lr_rsl_rl": alg_a.learning_rate, "lr_graphed": float(upd.lr)})
  assert abs(alg_a.learning_rate - float(upd.lr)) < 1e-9
  assert worst < 1e-5, worst


if __name__ == "__main__":
  test_update_matches_rsl_rl()
