"""PEARL meta-training (Rakelly et al. 2019, Algorithm 1) with the same factory-style
interface as ``train_rl2_ppo`` in RL2-Torch-Experiments.

Losses (paper Eq. 1-4 + Sec. 4.3 / 5):

* critic + encoder:  L_critic = E[(Q(s,a,z) - (scale*r + (1-d) * gamma * V_bar(s', z_bar)))^2]
                     + beta * KL(q(z|c) || N(0, I));  gradients of the critic loss flow into q_phi
* value:             V(s, z_bar) regresses E_{a~pi}[min Q(s,a,z_bar) - log pi(a|s,z_bar)]
* actor:             SAC actor loss with z_bar = stop_gradient(z)
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from agents.pearl_agent import PEARLAgent, kl_to_prior
from utils.buffer import MultiTaskReplay
from utils.distributions import TanhNormal
from utils.evaluation import run_meta_eval
from utils.rollout import SingleTaskEnv, collect_steps


@dataclass
class PEARLConfig:
    """Hyper-parameters.  Values marked (authors) are the defaults of the authors'
    public code as remembered by the implementer; they are NOT given in the paper PDF."""
    num_train_tasks: int = 20
    num_eval_tasks: int = 16
    iterations: int = 200
    tasks_per_iter: int = 5            # tasks whose data is refreshed per iteration
    steps_initial: int = 2000          # prior-z steps per task before training
    steps_prior: int = 400             # per refreshed task, z ~ prior, goes to both buffers
    steps_posterior: int = 600         # per refreshed task, z ~ q(z|c), goes to both buffers
    steps_extra_rl: int = 400          # per refreshed task, z ~ q(z|c), RL buffer only
    train_steps_per_iter: int = 1000
    meta_batch: int = 16               # (authors) tasks per gradient step
    batch_size: int = 256              # (authors) RL batch per task
    context_batch_size: int = 100      # (authors) context transitions per task
    lr: float = 3e-4                   # (authors)
    gamma: float = 0.99
    tau: float = 0.005                 # (authors) target-V Polyak rate
    reward_scale: float = 5.0          # (authors) plays the role of 1/alpha in SAC
    kl_lambda: float = 0.1             # (authors) beta in Eq. 4; KL is summed over tasks
    mean_reg: float = 1e-3             # (authors) continuous policy regularisers
    std_reg: float = 1e-3
    rl_buffer_capacity: int = 1_000_000
    enc_buffer_capacity: int = 10_000
    eval_every: int = 10
    eval_trajectories: int = 3
    seed: int = 0


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, *x.shape[2:])


def pearl_update(agent: PEARLAgent, opts: Dict[str, torch.optim.Optimizer], ctx: Dict[str, torch.Tensor],
                 batch: Dict[str, torch.Tensor], cfg: PEARLConfig) -> Dict[str, float]:
    """One PEARL gradient step.

    Parameters
    ----------
    ctx : context batch, values (T, N, ...) from the encoder buffers
    batch : RL batch, values (T, B, ...) from the RL buffers
    opts : optimizers with keys ``encoder``, ``q``, ``vf``, ``policy``

    Returns
    -------
    dict of scalar diagnostics (floats).
    """
    T, B = batch["obs"].shape[:2]
    obs, act, rew, nobs, done = (batch[k] for k in ("obs", "act", "rew", "next_obs", "done"))

    # ---- inference network: z ~ q_phi(z | c), reparameterised (T, L)
    mu, var = agent.encoder(agent.make_context(ctx["obs"], ctx["act"], ctx["rew"], ctx["next_obs"]))
    z = mu + var.sqrt() * torch.randn_like(mu)
    z_b = z.unsqueeze(1).expand(T, B, -1)          # (T, B, L), keeps the graph for the critic loss
    z_bar = z_b.detach()                           # stop-gradient copy for actor / value

    # ---- critic + encoder (Eq. 4: Bellman error + beta * KL)
    with torch.no_grad():
        target = cfg.reward_scale * rew + (1.0 - done) * cfg.gamma * agent.v_forward(nobs, z_bar, target=True)
    q1, q2 = agent.q_forward(obs, act, z_b)
    if agent.is_discrete:                          # pick Q(s, a, z) for the taken action
        idx = act.long().unsqueeze(-1)
        q1, q2 = q1.gather(-1, idx), q2.gather(-1, idx)
    qf_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    kl = kl_to_prior(mu, var).sum()                # summed over tasks (authors' code)
    for k in ("encoder", "q"):
        opts[k].zero_grad(set_to_none=True)
    (qf_loss + cfg.kl_lambda * kl).backward()
    opts["encoder"].step()
    opts["q"].step()

    # ---- value and actor (z detached)
    info: Dict[str, float] = {}
    if agent.is_discrete:
        logits, _ = agent.policy_forward(obs, z_bar)
        logp_all = F.log_softmax(logits, dim=-1)
        probs = logp_all.exp()
        with torch.no_grad():
            qa, qb = agent.q_forward(obs, None, z_bar)
            q_min = torch.min(qa, qb)
        v_target = (probs.detach() * (q_min - logp_all.detach())).sum(-1, keepdim=True)
        policy_loss = (probs * (logp_all - q_min)).sum(-1).mean()
        info["entropy"] = float(-(probs * logp_all).sum(-1).mean().detach())
    else:
        mu_a, log_std = agent.policy_forward(obs, z_bar)
        dist = TanhNormal(mu_a, log_std.exp(), -1.0, 1.0)
        new_a, u = dist.rsample()
        logp = dist.log_prob_from_u(u).sum(-1, keepdim=True)
        qa, qb = agent.q_forward(obs, new_a, z_bar)
        q_min = torch.min(qa, qb)
        v_target = (q_min - logp).detach()
        policy_loss = (logp - q_min).mean() + cfg.mean_reg * mu_a.pow(2).mean() + cfg.std_reg * log_std.pow(2).mean()
        info["entropy"] = float(-logp.mean().detach())

    vf_loss = F.mse_loss(agent.v_forward(obs, z_bar), v_target)
    opts["vf"].zero_grad(set_to_none=True)
    vf_loss.backward()
    opts["vf"].step()

    opts["policy"].zero_grad(set_to_none=True)
    policy_loss.backward()
    opts["policy"].step()

    agent.soft_update_target(cfg.tau)
    info.update(qf_loss=float(qf_loss.detach()), vf_loss=float(vf_loss.detach()), policy_loss=float(policy_loss.detach()),
                kl=float(kl.detach()) / T, z_std=float(var.sqrt().mean().detach()))
    return info


def train_pearl(
    model_factory: Callable[[], PEARLAgent],
    sample_tasks_train: Callable[[int], List[dict]],
    make_envs_train: Callable,
    sample_tasks_eval: Callable[[int], List[dict]],
    make_envs_eval: Callable,
    action_dim: int,
    is_discrete: bool,
    action_low: Optional[float] = None,
    action_high: Optional[float] = None,
    save_path: Optional[str] = None,
    device: Optional[str] = None,
    log_fn: Callable[[str], None] = print,
    **overrides,
) -> PEARLAgent:
    """Meta-train PEARL.

    The arguments mirror ``train_rl2_ppo``: factories that sample tasks and build
    vector envs, plus action metadata.  Any field of :class:`PEARLConfig` can be
    passed as a keyword override (e.g. ``iterations=50``).

    The training task set is sampled **once** (``cfg.num_train_tasks`` tasks) and
    kept fixed, as in the paper; evaluation uses ``sample_tasks_eval``.

    Returns
    -------
    PEARLAgent (in train mode).  ``save_path`` always holds the best policy so far (highest mean
    return on the last evaluation trajectory); it is overwritten only on improvement.
    """
    cfg = PEARLConfig(**overrides)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    agent = model_factory().to(dev)
    agent.train()

    train_tasks = sample_tasks_train(cfg.num_train_tasks)
    envs = [SingleTaskEnv(make_envs_train([t]), is_discrete, action_low, action_high) for t in train_tasks]
    replay = MultiTaskReplay(cfg.num_train_tasks, agent.obs_dim, action_dim, is_discrete,
                             cfg.rl_buffer_capacity, cfg.enc_buffer_capacity)

    opts = {
        "encoder": torch.optim.Adam(agent.encoder.parameters(), lr=cfg.lr),
        "q": torch.optim.Adam(list(agent.q1.parameters()) + list(agent.q2.parameters()), lr=cfg.lr),
        "vf": torch.optim.Adam(agent.vf.parameters(), lr=cfg.lr),
        "policy": torch.optim.Adam(agent.policy.parameters(), lr=cfg.lr),
    }

    def refresh_normalizers(task: int, n_new: int):
        """Feed the newest ``n_new`` transitions of a task to the running normalisers."""
        buf = replay.rl[task]
        idx = (np.arange(buf.ptr - n_new, buf.ptr)) % buf.capacity
        agent.normalizer_update(torch.as_tensor(buf.obs[idx], device=dev), torch.as_tensor(buf.rew[idx], device=dev))

    def collect(task: int, n: int, posterior: bool, to_enc: bool):
        if n <= 0:
            return
        collect_steps(envs[task], agent, replay, task, n, dev, use_posterior=posterior,
                      context_batch_size=cfg.context_batch_size, add_to_enc=to_enc)
        refresh_normalizers(task, min(n, replay.rl[task].size))

    t0 = time.time()
    log_fn(f"[init] collecting {cfg.steps_initial} prior steps for each of {cfg.num_train_tasks} tasks")
    for k in range(cfg.num_train_tasks):                       # Alg. 1 line 2-3: initialise buffers
        collect(k, cfg.steps_initial, posterior=False, to_enc=True)

    best = -np.inf
    for it in range(1, cfg.iterations + 1):
        # ---- data collection (Alg. 1 lines 4-14)
        for k in np.random.choice(cfg.num_train_tasks, min(cfg.tasks_per_iter, cfg.num_train_tasks), replace=False):
            replay.clear_enc(int(k))
            collect(int(k), cfg.steps_prior, posterior=False, to_enc=True)
            collect(int(k), cfg.steps_posterior, posterior=True, to_enc=True)
            collect(int(k), cfg.steps_extra_rl, posterior=True, to_enc=False)

        # ---- meta-training steps (Alg. 1 lines 15-23)
        stats: Dict[str, float] = {}
        for _ in range(cfg.train_steps_per_iter):
            tasks = np.random.choice(cfg.num_train_tasks, cfg.meta_batch, replace=cfg.meta_batch > cfg.num_train_tasks)
            ctx = replay.sample_context(tasks, cfg.context_batch_size, dev)
            batch = replay.sample_rl(tasks, cfg.batch_size, dev)
            info = pearl_update(agent, opts, ctx, batch, cfg)
            for k, v in info.items():
                stats[k] = stats.get(k, 0.0) + v / cfg.train_steps_per_iter

        msg = f"[it {it:4d}/{cfg.iterations}] " + " ".join(f"{k}={v:.3g}" for k, v in stats.items()) + f" ({time.time() - t0:.0f}s)"
        if it % cfg.eval_every == 0 or it == cfg.iterations:
            ev = run_meta_eval(agent, make_envs_eval, sample_tasks_eval(cfg.num_eval_tasks), dev, is_discrete,
                               action_low, action_high, num_trajectories=cfg.eval_trajectories)
            msg += " | eval return per trajectory: " + " ".join(f"{m:.1f}" for m in ev["mean"])
            if save_path and ev["mean"][-1] > best:                    # best last-trajectory eval return so far
                best = float(ev["mean"][-1])
                save_checkpoint(agent, cfg, save_path)
                msg += f" | new best, saved to {os.path.basename(save_path)}"
        log_fn(msg)
    return agent


def save_checkpoint(agent: PEARLAgent, cfg: PEARLConfig, path: str) -> None:
    """Save ``{"model": state_dict, "config": cfg}``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"model": agent.state_dict(), "config": asdict(cfg)}, path)


def load_checkpoint(agent: PEARLAgent, path: str, map_location="cpu") -> PEARLAgent:
    """Load weights saved by :func:`save_checkpoint` into an agent built by the same factory."""
    agent.load_state_dict(torch.load(path, map_location=map_location)["model"])
    return agent