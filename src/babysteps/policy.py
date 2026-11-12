"""Portable policy format used by every track.

The evaluator never imports training code. Each checkpoint written during a run
carries a self-contained description of the deterministic actor:

  obs -> (obs - mean) / denom -> [Linear -> act]* -> Linear -> action

This keeps the evaluation identical across tracks (PPO, SAC, custom networks)
and lets the CPU sim-to-sim check run the policy in plain numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

FORMAT = "bs-policy-v1"

_ACTIVATIONS_NP = {
  "elu": lambda x: np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0))),
  "relu": lambda x: np.maximum(x, 0.0),
  "tanh": np.tanh,
  "silu": lambda x: x / (1.0 + np.exp(-x)),
  "identity": lambda x: x,
}

_ACTIVATIONS_T = {
  "elu": torch.nn.functional.elu,
  "relu": torch.relu,
  "tanh": torch.tanh,
  "silu": torch.nn.functional.silu,
  "identity": lambda x: x,
}


@dataclass
class PortablePolicy:
  obs_mean: np.ndarray
  obs_denom: np.ndarray
  weights: list[np.ndarray]
  biases: list[np.ndarray]
  activation: str = "elu"
  output_activation: str = "identity"
  meta: dict | None = None

  @property
  def obs_dim(self) -> int:
    return int(self.weights[0].shape[1])

  @property
  def act_dim(self) -> int:
    return int(self.weights[-1].shape[0])

  # numpy path (CPU sim-to-sim) -------------------------------------------------
  def act_np(self, obs: np.ndarray) -> np.ndarray:
    x = (obs.astype(np.float64) - self.obs_mean) / self.obs_denom
    act = _ACTIVATIONS_NP[self.activation]
    n = len(self.weights)
    for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
      x = x @ w.T + b
      if i < n - 1:
        x = act(x)
    return _ACTIVATIONS_NP[self.output_activation](x)

  # torch path (GPU evaluation) -------------------------------------------------
  def to_torch(self, device: str) -> "TorchPolicy":
    return TorchPolicy(self, device)

  def to_dict(self) -> dict:
    return {
      "format": FORMAT,
      "obs_mean": torch.as_tensor(self.obs_mean, dtype=torch.float32),
      "obs_denom": torch.as_tensor(self.obs_denom, dtype=torch.float32),
      "weights": [torch.as_tensor(w, dtype=torch.float32) for w in self.weights],
      "biases": [torch.as_tensor(b, dtype=torch.float32) for b in self.biases],
      "activation": self.activation,
      "output_activation": self.output_activation,
      "meta": self.meta or {},
    }

  @staticmethod
  def from_dict(d: dict) -> "PortablePolicy":
    if d.get("format") != FORMAT:
      raise ValueError(f"unknown policy format: {d.get('format')}")
    f = lambda t: t.detach().cpu().double().numpy()  # noqa: E731
    return PortablePolicy(
      obs_mean=f(d["obs_mean"]),
      obs_denom=f(d["obs_denom"]),
      weights=[f(w) for w in d["weights"]],
      biases=[f(b) for b in d["biases"]],
      activation=d["activation"],
      output_activation=d.get("output_activation", "identity"),
      meta=d.get("meta", {}),
    )


class TorchPolicy(torch.nn.Module):
  def __init__(self, p: PortablePolicy, device: str):
    super().__init__()
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    self.register_buffer("mean", t(p.obs_mean))
    self.register_buffer("denom", t(p.obs_denom))
    self.ws = [t(w) for w in p.weights]
    self.bs = [t(b) for b in p.biases]
    self.act = _ACTIVATIONS_T[p.activation]
    self.out_act = _ACTIVATIONS_T[p.output_activation]

  @torch.no_grad()
  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    x = (obs - self.mean) / self.denom
    n = len(self.ws)
    for i, (w, b) in enumerate(zip(self.ws, self.bs, strict=True)):
      x = torch.addmm(b, x, w.T)
      if i < n - 1:
        x = self.act(x)
    return self.out_act(x)


def from_rsl_rl_actor(actor: torch.nn.Module, activation: str = "elu") -> PortablePolicy:
  """Extract a PortablePolicy from an rsl_rl ``MLPModel`` (Gaussian head, mean output)."""
  actor = getattr(actor, "_orig_mod", actor)  # unwrap torch.compile
  norm = actor.obs_normalizer
  linears = [m for m in actor.mlp if isinstance(m, torch.nn.Linear)]
  obs_dim = linears[0].in_features
  if isinstance(norm, torch.nn.Identity):
    mean = np.zeros(obs_dim)
    denom = np.ones(obs_dim)
  else:
    mean = norm._mean.detach().squeeze(0).cpu().double().numpy()
    denom = (norm._std.detach().squeeze(0) + norm.eps).cpu().double().numpy()
  return PortablePolicy(
    obs_mean=mean,
    obs_denom=denom,
    weights=[m.weight.detach().cpu().double().numpy() for m in linears],
    biases=[m.bias.detach().cpu().double().numpy() for m in linears],
    activation=activation,
  )


def save(policy: PortablePolicy, path: str | Path) -> None:
  torch.save(policy.to_dict(), path)


def load(path: str | Path) -> PortablePolicy:
  return PortablePolicy.from_dict(torch.load(path, map_location="cpu", weights_only=False))
