"""PEARL networks: probabilistic context encoder + SAC actor / critics.

Notation follows Rakelly et al. (2019), "Efficient Off-Policy Meta-Reinforcement
Learning via Probabilistic Context Variables" (arXiv 1903.08254):

* ``c``  : context, a set of transitions ``(s, a, r, s')`` from one task.
* ``q_phi(z | c) = prod_n Psi_phi(z | c_n)`` : permutation-invariant posterior,
  each factor is a diagonal Gaussian ``N(f_mu(c_n), f_sigma(c_n))`` (Eq. 3-4 / Sec 4.1).
* ``pi_theta(a | s, z)``, ``Q_theta(s, a, z)``, ``V_psi(s, z)`` : SAC networks conditioned on z.

Two action-space variants share one class:

* continuous (Pendulum, HalfCheetah): tanh-Gaussian policy, ``Q(s, a, z) -> R``.
* discrete (CartPole): categorical policy, ``Q(s, z) -> R^{|A|}`` (SAC-Discrete style).

Shape conventions (``T`` = number of tasks in a meta batch, ``N`` = context size,
``B`` = RL batch size, ``L`` = latent dim)::

    context          (T, N, ctx_dim)     ctx_dim = 2*obs_dim + act_feat_dim + 1
    z, mu, var       (T, L)
    obs / next_obs   (T, B, obs_dim)
    action           (T, B, action_dim)  continuous, in [-1, 1]
                     (T, B)              discrete, int64
"""
from __future__ import annotations

import copy
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.nn_tools import mlp
from utils.distributions import TanhNormal, make_action_dist

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class ContextEncoder(nn.Module):
    """Permutation-invariant probabilistic encoder ``q_phi(z | c)``.

    Every transition ``c_n`` is mapped by one MLP to ``(mu_n, sigma_n^2)``.  The
    posterior is the (normalised) product of the Gaussian factors::

        var = 1 / sum_n (1 / var_n),     mu = var * sum_n (mu_n / var_n)

    which is invariant to the order of the transitions and to nothing else.
    With an empty context the posterior is the prior ``N(0, I)``.

    Parameters
    ----------
    ctx_dim : int
        Size of one transition feature vector.
    latent_dim : int
        Dimension ``L`` of z.
    hidden_dims : Sequence[int]
        Hidden widths of the per-transition MLP.
    """

    def __init__(self, ctx_dim: int, latent_dim: int, hidden_dims: Sequence[int] = (200, 200, 200)):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = mlp(ctx_dim, 2 * latent_dim, hidden_dims, activation=nn.ReLU, last_std=1.0)

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the posterior parameters.

        Parameters
        ----------
        context : Tensor, shape (..., N, ctx_dim)
            ``N`` transitions per task (any number of leading batch dims).

        Returns
        -------
        mu, var : Tensor, each shape (..., L)
            Mean and variance of the diagonal Gaussian ``q(z | c)``.
        """
        params = self.net(context)                                   # (..., N, 2L)
        mu_n = params[..., : self.latent_dim]                         # (..., N, L)
        var_n = F.softplus(params[..., self.latent_dim:]).clamp(min=1e-7)
        return product_of_gaussians(mu_n, var_n)


def product_of_gaussians(mu_n: torch.Tensor, var_n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine ``N`` diagonal Gaussian factors into one Gaussian.

    Parameters
    ----------
    mu_n, var_n : Tensor, shape (..., N, L)
        Per-factor mean / variance (variance must be > 0).

    Returns
    -------
    mu, var : Tensor, shape (..., L)
    """
    precision = 1.0 / var_n
    var = 1.0 / precision.sum(dim=-2)
    mu = var * (mu_n * precision).sum(dim=-2)
    return mu, var


def kl_to_prior(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """``KL( N(mu, var) || N(0, I) )`` summed over the latent dimension.

    Parameters
    ----------
    mu, var : Tensor, shape (..., L)

    Returns
    -------
    Tensor, shape (...)
    """
    return 0.5 * (var + mu.pow(2) - 1.0 - var.log()).sum(dim=-1)


class PEARLAgent(nn.Module):
    """Container for the encoder, actor, twin critics, V and target V.

    Parameters
    ----------
    obs_dim, action_dim : int
        ``action_dim`` is the number of discrete actions if ``is_discrete``.
    is_discrete : bool
        Categorical policy / per-action Q head if True; tanh-Gaussian otherwise.
    latent_dim : int
        Size of z (authors' code: 5; from memory, not stated in the paper PDF).
    hidden_dims : Sequence[int]
        Hidden widths of policy, Q and V networks.
    encoder_hidden_dims : Sequence[int]
        Hidden widths of the context encoder.
    obs_norm, rew_norm : nn.Module | None
        Optional ``RunningMeanStd1DNormalizer`` for observations / rewards (reward
        normalisation is applied to the *context only*; the Bellman target uses
        the raw reward times ``reward_scale``).
    use_next_obs_in_context : bool
        Whether the context transition contains ``s'`` (the paper uses full
        transitions ``(s, a, r, s')``).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        is_discrete: bool,
        latent_dim: int = 5,
        hidden_dims: Sequence[int] = (300, 300, 300),
        encoder_hidden_dims: Sequence[int] = (200, 200, 200),
        obs_norm: Optional[nn.Module] = None,
        rew_norm: Optional[nn.Module] = None,
        use_next_obs_in_context: bool = True,
    ):
        super().__init__()
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.is_discrete = is_discrete
        self.latent_dim = latent_dim
        self.use_next_obs_in_context = use_next_obs_in_context
        self.obs_norm, self.rew_norm = obs_norm, rew_norm

        self.act_feat_dim = action_dim  # one-hot (discrete) or the action itself
        ctx_dim = obs_dim + self.act_feat_dim + 1 + (obs_dim if use_next_obs_in_context else 0)
        self.ctx_dim = ctx_dim
        self.encoder = ContextEncoder(ctx_dim, latent_dim, encoder_hidden_dims)

        relu = nn.ReLU
        if is_discrete:
            self.policy = mlp(obs_dim + latent_dim, action_dim, hidden_dims, relu, last_std=0.01)
            self.q1 = mlp(obs_dim + latent_dim, action_dim, hidden_dims, relu, last_std=1.0)
            self.q2 = mlp(obs_dim + latent_dim, action_dim, hidden_dims, relu, last_std=1.0)
        else:
            self.policy = mlp(obs_dim + latent_dim, 2 * action_dim, hidden_dims, relu, last_std=0.01)
            self.q1 = mlp(obs_dim + action_dim + latent_dim, 1, hidden_dims, relu, last_std=1.0)
            self.q2 = mlp(obs_dim + action_dim + latent_dim, 1, hidden_dims, relu, last_std=1.0)
        self.vf = mlp(obs_dim + latent_dim, 1, hidden_dims, relu, last_std=1.0)
        self.target_vf = copy.deepcopy(self.vf)
        for p in self.target_vf.parameters():
            p.requires_grad_(False)

        # Current task belief, used while acting.  Shapes (1, L).
        self.register_buffer("z_mu", torch.zeros(1, latent_dim))
        self.register_buffer("z_var", torch.ones(1, latent_dim))
        self.register_buffer("z", torch.zeros(1, latent_dim))

    # ------------------------------------------------------------------ utils
    def encoder_parameters(self):
        return self.encoder.parameters()

    def actor_parameters(self):
        return self.policy.parameters()

    def normalizer_update(self, obs: torch.Tensor, rew: torch.Tensor) -> None:
        """Update running statistics.

        Parameters
        ----------
        obs : Tensor (..., obs_dim); rew : Tensor (..., 1)
        """
        if self.obs_norm is not None:
            self.obs_norm.update(obs)
        if self.rew_norm is not None:
            self.rew_norm.update(rew)

    def norm_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalise observations (identity if no normaliser). Shape preserved."""
        return obs if self.obs_norm is None else self.obs_norm(obs)

    def make_context(self, obs, act, rew, next_obs) -> torch.Tensor:
        """Build context feature vectors.

        Parameters
        ----------
        obs, next_obs : Tensor, shape (..., obs_dim)
        act : Tensor, shape (...,) int64 (discrete) or (..., action_dim) float
        rew : Tensor, shape (..., 1)

        Returns
        -------
        Tensor, shape (..., ctx_dim)
        """
        a = F.one_hot(act.long(), self.action_dim).float() if self.is_discrete else act
        r = rew if self.rew_norm is None else self.rew_norm(rew)
        parts = [self.norm_obs(obs), a, r]
        if self.use_next_obs_in_context:
            parts.append(self.norm_obs(next_obs))
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------- task belief
    def clear_z(self) -> None:
        """Reset the belief to the prior ``N(0, I)`` and draw ``z`` from it."""
        self.z_mu.zero_()
        self.z_var.fill_(1.0)
        self.sample_z()

    @torch.no_grad()
    def sample_z(self) -> torch.Tensor:
        """Draw ``z ~ q(z | c)`` for acting. Returns (1, L)."""
        self.z.copy_(self.z_mu + self.z_var.sqrt() * torch.randn_like(self.z_mu))
        return self.z

    @torch.no_grad()
    def infer_posterior(self, context: torch.Tensor) -> None:
        """Update the belief from context of shape (1, N, ctx_dim) or (N, ctx_dim)."""
        if context.dim() == 2:
            context = context.unsqueeze(0)
        mu, var = self.encoder(context)
        self.z_mu.copy_(mu)
        self.z_var.copy_(var)
        self.sample_z()

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """Select an action for one observation using the current ``z``.

        Parameters
        ----------
        obs : Tensor, shape (obs_dim,)

        Returns
        -------
        np.ndarray : int64 scalar array (discrete) or float32 (action_dim,) in [-1, 1].
        """
        x = torch.cat([self.norm_obs(obs.unsqueeze(0)), self.z], dim=-1)
        out = self.policy(x)
        if self.is_discrete:
            a = out.argmax(-1) if deterministic else torch.distributions.Categorical(logits=out).sample()
            return a[0].cpu().numpy()
        mu, log_std = out.chunk(2, dim=-1)
        if deterministic:
            a = torch.tanh(mu)
        else:
            a, _ = TanhNormal(mu, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp(), -1.0, 1.0).sample()
        return a[0].cpu().numpy()

    # --------------------------------------------- batched SAC heads (training)
    def policy_forward(self, obs: torch.Tensor, z: torch.Tensor):
        """Policy outputs for a batch.

        Parameters
        ----------
        obs : (T, B, obs_dim) raw observations; z : (T, B, L)

        Discrete  -> ``(logits, None)`` with logits (T, B, |A|)
        Continuous-> ``(mu, log_std)`` each (T, B, action_dim)
        """
        out = self.policy(torch.cat([self.norm_obs(obs), z], dim=-1))
        if self.is_discrete:
            return out, None
        mu, log_std = out.chunk(2, dim=-1)
        return mu, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def q_forward(self, obs: torch.Tensor, act: Optional[torch.Tensor], z: torch.Tensor):
        """Twin Q values.

        Discrete: returns two (T, B, |A|) tensors (``act`` ignored).
        Continuous: ``act`` (T, B, action_dim) in [-1, 1]; two (T, B, 1) tensors.
        """
        o = self.norm_obs(obs)
        if self.is_discrete:
            x = torch.cat([o, z], dim=-1)
        else:
            x = torch.cat([o, act, z], dim=-1)
        return self.q1(x), self.q2(x)

    def v_forward(self, obs: torch.Tensor, z: torch.Tensor, target: bool = False) -> torch.Tensor:
        """``V(s, z)`` or ``V_bar(s, z)``; returns (T, B, 1)."""
        net = self.target_vf if target else self.vf
        return net(torch.cat([self.norm_obs(obs), z], dim=-1))

    @torch.no_grad()
    def soft_update_target(self, tau: float) -> None:
        """Polyak-average ``V -> V_bar``."""
        for p, tp in zip(self.vf.parameters(), self.target_vf.parameters()):
            tp.mul_(1.0 - tau).add_(tau * p)
