"""Unit tests: ``PYTHONPATH=. pytest tests``."""
import os, sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from agents.pearl_agent import PEARLAgent, ContextEncoder, product_of_gaussians, kl_to_prior
from train import PEARLConfig, pearl_update
from utils.buffer import MultiTaskReplay
from utils.rollout import SingleTaskEnv, collect_steps
from utils.evaluation import run_meta_eval


def test_product_of_gaussians_matches_closed_form():
    mu1, v1, mu2, v2 = 1.0, 2.0, -3.0, 0.5
    mu, var = product_of_gaussians(torch.tensor([[[mu1], [mu2]]]), torch.tensor([[[v1], [v2]]]))
    exp_var = 1 / (1 / v1 + 1 / v2)
    assert var.item() == pytest.approx(exp_var)
    assert mu.item() == pytest.approx(exp_var * (mu1 / v1 + mu2 / v2))


def test_encoder_permutation_invariant_and_shapes():
    enc = ContextEncoder(7, 5, (16, 16))
    c = torch.randn(3, 11, 7)
    mu, var = enc(c)
    assert mu.shape == var.shape == (3, 5) and (var > 0).all()
    mu2, var2 = enc(c[:, torch.randperm(11)])
    assert torch.allclose(mu, mu2, atol=1e-5) and torch.allclose(var, var2, atol=1e-5)


def test_more_context_shrinks_variance():
    enc = ContextEncoder(4, 3, (16,))
    c = torch.randn(1, 64, 4)
    assert enc(c)[1].mean() < enc(c[:, :4])[1].mean()


def test_kl_zero_at_prior():
    assert kl_to_prior(torch.zeros(2, 5), torch.ones(2, 5)).abs().max() < 1e-6


def _agent(discrete):
    return PEARLAgent(4, 2 if discrete else 1, discrete, latent_dim=3, hidden_dims=(32, 32), encoder_hidden_dims=(32,))


def _fill(replay, discrete, n=200):
    for t in range(replay.num_tasks):
        for _ in range(n):
            a = np.random.randint(2) if discrete else np.random.uniform(-1, 1, 1).astype(np.float32)
            replay.add(t, np.random.randn(4).astype(np.float32), a, np.random.rand(), np.random.randn(4).astype(np.float32), False)


@pytest.mark.parametrize("discrete", [True, False])
def test_update_gradient_flow_and_loss_finite(discrete):
    agent, cfg = _agent(discrete), PEARLConfig(meta_batch=3, batch_size=16, context_batch_size=8)
    replay = MultiTaskReplay(3, 4, 2 if discrete else 1, discrete, 1000, 1000)
    _fill(replay, discrete)
    opts = {"encoder": torch.optim.Adam(agent.encoder.parameters()),
            "q": torch.optim.Adam(list(agent.q1.parameters()) + list(agent.q2.parameters())),
            "vf": torch.optim.Adam(agent.vf.parameters()), "policy": torch.optim.Adam(agent.policy.parameters())}
    enc0 = [p.clone() for p in agent.encoder.parameters()]
    tv0 = [p.clone() for p in agent.target_vf.parameters()]
    info = pearl_update(agent, opts, replay.sample_context([0, 1, 2], 8, "cpu"), replay.sample_rl([0, 1, 2], 16, "cpu"), cfg)
    assert all(np.isfinite(v) for v in info.values())
    assert any((a != b).any() for a, b in zip(enc0, agent.encoder.parameters())), "encoder must learn from the critic loss"
    assert any((a != b).any() for a, b in zip(tv0, agent.target_vf.parameters())), "target V must be Polyak-updated"


def test_policy_loss_does_not_move_encoder():
    """z is detached for the actor / value losses (Sec. 4.3)."""
    agent = _agent(True)
    z = torch.randn(2, 5, 3, requires_grad=True)
    logits, _ = agent.policy_forward(torch.randn(2, 5, 4), z.detach())
    logits.sum().backward()
    assert z.grad is None


def test_buffer_enc_vs_rl_and_shapes():
    replay = MultiTaskReplay(2, 4, 1, False, 50, 10)
    _fill(replay, False, 30)
    assert replay.rl[0].size == 30 and replay.enc[0].size == 10  # ring buffer keeps only recent
    replay.clear_enc(0)
    assert replay.enc[0].size == 0 and replay.rl[0].size == 30
    b = replay.sample_rl([0, 1], 7, "cpu")
    assert b["obs"].shape == (2, 7, 4) and b["act"].shape == (2, 7, 1) and b["rew"].shape == (2, 7, 1)


def test_env_runner_resets_cleanly_after_done():
    import cartpole_wind as c
    env = SingleTaskEnv(c.make_envs([{"wind_mag": 5.0}]), True)
    for _ in range(3):
        obs = env.reset()
        assert np.abs(obs).max() < 0.06            # CartPole reset range, i.e. not a stale terminal obs
        done, n = False, 0
        while not done:
            _, r, term, trunc = env.step(np.array(1))
            done, n = term or trunc, n + 1
        assert n < 500 or trunc


def test_collect_and_eval_discrete():
    import cartpole_wind as c
    agent = c.cartpole_model_factory(hidden_dims=(32, 32))
    replay = MultiTaskReplay(1, 4, 2, True, 5000, 5000)
    env = SingleTaskEnv(c.make_envs([{"wind_mag": 1.0}]), True)
    out = collect_steps(env, agent, replay, 0, 120, "cpu", use_posterior=False)
    assert out["steps"] == 120 and replay.rl[0].size == 120 == replay.enc[0].size
    collect_steps(env, agent, replay, 0, 60, "cpu", use_posterior=True, add_to_enc=False)
    assert replay.rl[0].size == 180 and replay.enc[0].size == 120
    ev = run_meta_eval(agent, c.make_envs, [{"wind_mag": -2.0}, {"wind_mag": 2.0}], "cpu", True, num_trajectories=2)
    assert ev["returns"].shape == (2, 2)


def test_time_limit_is_not_a_terminal_flag():
    """done stored in the buffer must be termination only (so V_bar bootstraps at TimeLimit)."""
    import pendulum_wind as p
    agent = p.pendulum_model_factory(hidden_dims=(16,))
    replay = MultiTaskReplay(1, 3, 1, False, 1000, 1000)
    env = SingleTaskEnv(p.make_envs([{"wind_torque": 0.0}]), False, -2.0, 2.0)
    collect_steps(env, agent, replay, 0, 250, "cpu", use_posterior=False)   # > one 200-step episode
    assert replay.rl[0].done[:250].sum() == 0
