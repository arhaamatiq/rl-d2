import numpy as np
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from rl_d2 import policy as policy_io
from rl_d2.policy import PortablePolicy


def _random_policy(seed: int = 0) -> PortablePolicy:
  rng = np.random.default_rng(seed)
  dims = [7, 16, 16, 3]
  return PortablePolicy(
    obs_mean=rng.normal(size=7),
    obs_denom=rng.uniform(0.5, 2.0, size=7),
    weights=[rng.normal(size=(dims[i + 1], dims[i])) * 0.5 for i in range(3)],
    biases=[rng.normal(size=dims[i + 1]) * 0.1 for i in range(3)],
    activation="elu",
    meta={"note": "test"},
  )


def test_numpy_and_torch_agree():
  p = _random_policy()
  obs = np.random.default_rng(1).normal(size=(5, 7))
  a_np = p.act_np(obs)
  a_t = p.to_torch("cpu")(torch.as_tensor(obs, dtype=torch.float32)).numpy()
  assert a_np.shape == (5, 3)
  np.testing.assert_allclose(a_np, a_t, atol=1e-5)


def test_save_load_roundtrip(tmp_path):
  p = _random_policy()
  path = tmp_path / "p.pt"
  policy_io.save(p, path)
  q = policy_io.load(path)
  assert q.obs_dim == 7 and q.act_dim == 3
  assert q.meta == {"note": "test"}
  obs = np.random.default_rng(2).normal(size=(4, 7))
  np.testing.assert_allclose(p.act_np(obs), q.act_np(obs), atol=1e-5)


def make_actor(obs_dim: int = 7, act_dim: int = 3) -> MLPModel:
  obs = TensorDict({"actor": torch.randn(4, obs_dim)}, batch_size=[4])
  return MLPModel(
    obs,
    {"actor": ["actor"]},
    "actor",
    act_dim,
    hidden_dims=(16, 16),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
  )


def test_from_rsl_rl_actor_matches_model():
  torch.manual_seed(0)
  actor = make_actor()
  actor.obs_normalizer._mean.copy_(torch.randn(1, 7))
  actor.obs_normalizer._std.copy_(torch.rand(1, 7) + 0.5)
  actor.eval()
  obs = TensorDict({"actor": torch.randn(4, 7)}, batch_size=[4])
  with torch.no_grad():
    expected = actor(obs, stochastic_output=False).numpy()
  p = policy_io.from_rsl_rl_actor(actor)
  np.testing.assert_allclose(p.act_np(obs["actor"].numpy()), expected, atol=1e-5)
