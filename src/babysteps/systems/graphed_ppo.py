"""rsl_rl 5.4.2 ``PPO.update`` rewritten so the whole update is one CUDA graph.

Same math, same order of operations:
  * one random permutation per update, shared by all epochs
  * per-minibatch KL between the stored and current Gaussian, adaptive LR
    (/1.5 above 2x target, x1.5 below 0.5x target, clamped to [1e-5, 1e-2]),
    applied to *this* minibatch's step, as rsl_rl does
  * clipped surrogate, clipped value loss, entropy bonus
  * separate grad-norm clipping for actor and critic, then one Adam step

What changes is where the control flow lives: the adaptive-LR comparisons run on
the device with ``torch.where`` and the LR is a device tensor read by a
capturable Adam, so the update has no host round trips. Loss statistics are
accumulated on the device and read once per iteration.
``tests/test_graphed_ppo_parity.py`` runs this and rsl_rl's ``update()`` on the
same storage and checks parameters match.
"""

from __future__ import annotations

import torch


def make_inplace_normalizer(norm) -> None:
  """EmpiricalNormalization.update reassigns ``_std``; make it in-place so a
  captured graph keeps reading the live buffer. Numerically identical."""
  if not hasattr(norm, "_var"):
    return

  def update(x):
    if not norm.training:
      return
    if norm.until is not None and norm.count >= norm.until:
      return
    count_x = x.shape[0]
    norm.count += count_x
    rate = count_x / norm.count
    var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
    mean_x = torch.mean(x, dim=0, keepdim=True)
    delta_mean = mean_x - norm._mean
    norm._mean += rate * delta_mean
    norm._var += rate * (var_x - norm._var + delta_mean * (mean_x - norm._mean))
    norm._std.copy_(torch.sqrt(norm._var))

  norm.update = update


def make_capturable_gaussian(dist) -> None:
  """torch.normal(mean, std) validates std >= 0 with a host sync and cannot be
  captured. Reparameterized sampling draws from the same distribution."""

  def sample():
    d = dist._distribution
    return d.loc + d.scale * torch.randn_like(d.loc)

  dist.sample = sample


class GraphedPPOUpdate:
  def __init__(self, alg, *, fused_adam: bool = False, amp_dtype: torch.dtype | None = None):
    self.alg = alg
    st = alg.storage
    self.device = alg.device
    self.T, self.N = st.num_transitions_per_env, st.num_envs
    self.B = self.T * self.N
    self.M = alg.num_mini_batches
    self.E = alg.num_learning_epochs
    self.mb = self.B // self.M
    params = list(alg.actor.parameters()) + list(alg.critic.parameters())
    self.lr = torch.tensor(float(alg.learning_rate), device=self.device)
    kw = {"fused": True} if fused_adam else {"foreach": True}
    self.opt = torch.optim.Adam(params, lr=self.lr, capturable=True, **kw)
    alg.optimizer = self.opt
    self.amp_dtype = amp_dtype
    # static views / buffers
    self.adv = torch.zeros(self.T, self.N, 1, device=self.device)
    self.stats = torch.zeros(5, device=self.device)  # value, surrogate, entropy, kl, lr_sum
    self.perm_override: torch.Tensor | None = None  # tests only

  def snapshot(self) -> dict:
    a = self.alg
    return {
      "actor": {k: v.detach().clone() for k, v in a.actor.state_dict().items()},
      "critic": {k: v.detach().clone() for k, v in a.critic.state_dict().items()},
      "lr": self.lr.detach().clone(),
    }

  def restore(self, snap: dict) -> None:
    a = self.alg
    with torch.no_grad():
      for k, v in a.actor.state_dict().items():
        v.copy_(snap["actor"][k])
      for k, v in a.critic.state_dict().items():
        v.copy_(snap["critic"][k])
      self.lr.copy_(snap["lr"])
      for state in self.opt.state.values():
        for t in state.values():
          if torch.is_tensor(t):
            t.zero_()

  def body(self, stages: tuple[str, ...] = ("fwd", "bwd", "opt")) -> None:
    """Full update by default. ``stages`` exists only for the component benchmark
    (forward+loss / +backward / +optimizer timed separately)."""
    a = self.alg
    st = a.storage
    actor, critic = a.actor, a.critic
    obs = st.observations.flatten(0, 1)
    actions = st.actions.flatten(0, 1)
    values = st.values.flatten(0, 1)
    returns = st.returns.flatten(0, 1)
    old_logp = st.actions_log_prob.flatten(0, 1)
    adv = self.adv.flatten(0, 1)
    old_mu, old_sigma = (p.flatten(0, 1) for p in st.distribution_params)
    perm = self.perm_override if self.perm_override is not None else torch.argsort(
      torch.rand(self.M * self.mb, device=self.device)
    )
    dkl = a.desired_kl
    self.stats.zero_()
    for _epoch in range(self.E):
      for i in range(self.M):
        idx = perm[i * self.mb : (i + 1) * self.mb]
        ob = {k: obs[k][idx] for k in obs.keys()}
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
          actor(ob, stochastic_output=True)
          value = critic(ob)
        logp = actor.get_output_log_prob(actions[idx])
        mu, sigma = actor.output_distribution_params
        entropy = actor.output_entropy
        value = value.float()
        if a.desired_kl is not None and a.schedule == "adaptive":
          with torch.no_grad():
            kl = actor.get_kl_divergence((old_mu[idx], old_sigma[idx]), (mu.float(), sigma.float()))
            kl_mean = kl.mean()
            down = kl_mean > dkl * 2.0
            up = (kl_mean < dkl / 2.0) & (kl_mean > 0.0)
            new_lr = torch.where(
              down,
              torch.clamp(self.lr / 1.5, min=1e-5),
              torch.where(up, torch.clamp(self.lr * 1.5, max=1e-2), self.lr),
            )
            self.lr.copy_(new_lr)
            self.stats[3] += kl_mean
        ratio = torch.exp(logp - torch.squeeze(old_logp[idx]))
        adv_b = torch.squeeze(adv[idx])
        surr = -adv_b * ratio
        surr_c = -adv_b * torch.clamp(ratio, 1.0 - a.clip_param, 1.0 + a.clip_param)
        surrogate_loss = torch.max(surr, surr_c).mean()
        v_old, ret = values[idx], returns[idx]
        if a.use_clipped_value_loss:
          v_clip = v_old + (value - v_old).clamp(-a.clip_param, a.clip_param)
          value_loss = torch.max((value - ret).pow(2), (v_clip - ret).pow(2)).mean()
        else:
          value_loss = (ret - value).pow(2).mean()
        loss = surrogate_loss + a.value_loss_coef * value_loss - a.entropy_coef * entropy.mean()
        if "bwd" in stages:
          self.opt.zero_grad(set_to_none=False)
          loss.backward()
        if "opt" in stages:
          torch.nn.utils.clip_grad_norm_(actor.parameters(), a.max_grad_norm)
          torch.nn.utils.clip_grad_norm_(critic.parameters(), a.max_grad_norm)
          self.opt.step()
        with torch.no_grad():
          self.stats[0] += value_loss.detach()
          self.stats[1] += surrogate_loss.detach()
          self.stats[2] += entropy.mean().detach()
          self.stats[4] += self.lr
    self.stats /= float(self.E * self.M)
