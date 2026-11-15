"""Single-task environment runner and data collection (PEARL Algorithm 1, lines 5-14)."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch


class SingleTaskEnv:
    """Thin wrapper around a ``make_envs([task])`` vector env with one sub-env.

    The experiments' ``make_envs`` factories return a ``SyncVectorEnv``; PEARL
    collects data one task at a time, so we use sub-env 0 only.  Episodes are
    always started with ``reset()`` (which also clears gymnasium's NEXT_STEP
    autoreset flag), so no dummy transitions are ever produced.

    Parameters
    ----------
    vec_env : gymnasium.vector.SyncVectorEnv
        Vector env with exactly one sub-env.
    is_discrete : bool
    action_low, action_high : float | array-like | None
        Bounds used to map a policy action in [-1, 1] to the env range.
    """

    def __init__(self, vec_env, is_discrete: bool, action_low=None, action_high=None):
        assert vec_env.num_envs == 1, "SingleTaskEnv expects a vector env with one sub-env"
        self.env, self.is_discrete = vec_env, is_discrete
        if not is_discrete:
            lo, hi = np.asarray(action_low, np.float32), np.asarray(action_high, np.float32)
            self._scale, self._bias = (hi - lo) / 2.0, (hi + lo) / 2.0

    def reset(self) -> np.ndarray:
        """Start a new episode. Returns obs of shape (obs_dim,)."""
        obs, _ = self.env.reset()
        return obs[0].astype(np.float32)

    def step(self, action: np.ndarray):
        """Step with a policy action.

        Parameters
        ----------
        action : int64 scalar (discrete) or float (action_dim,) in [-1, 1].

        Returns
        -------
        next_obs (obs_dim,), reward float, terminated bool, truncated bool
        """
        if self.is_discrete:
            env_a = np.asarray([int(action)])
        else:
            env_a = (self._bias + self._scale * np.clip(action, -1.0, 1.0))[None].astype(np.float32)
        obs, rew, term, trunc, _ = self.env.step(env_a)
        return obs[0].astype(np.float32), float(rew[0]), bool(term[0]), bool(trunc[0])

    def close(self):
        self.env.close()


def _to_t(x, device, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=device)


def collect_steps(
    env: SingleTaskEnv,
    agent,
    replay,
    task: int,
    num_steps: int,
    device,
    use_posterior: bool,
    context_batch_size: int = 100,
    add_to_enc: bool = True,
    update_posterior_every: int = 1,
    max_trajectories: Optional[int] = None,
) -> Dict[str, float]:
    """Collect ``num_steps`` environment steps for one task.

    * ``use_posterior=False`` -> ``z`` is drawn from the prior ``N(0, I)`` for every
      trajectory (exploration; Algorithm 1 line 8).
    * ``use_posterior=True``  -> after every ``update_posterior_every`` finished
      trajectories, a context batch is sampled from the task's encoder buffer, the
      posterior ``q(z|c)`` is recomputed and ``z`` is re-sampled (Algorithm 1 lines 9-10).

    The final trajectory may be cut when ``num_steps`` is reached.

    Returns
    -------
    dict with ``steps``, ``trajectories`` and ``mean_return`` (undiscounted, finished
    trajectories only; NaN if none finished).
    """
    returns: List[float] = []
    steps = trajs = 0
    agent.eval()
    if not use_posterior:
        agent.clear_z()
    while steps < num_steps:
        obs = env.reset()
        ep_ret, done = 0.0, False
        while not done and steps < num_steps:
            a = agent.act(_to_t(obs, device), deterministic=False)
            next_obs, rew, term, trunc = env.step(a)
            replay.add(task, obs, a, rew, next_obs, term, to_enc=add_to_enc)
            obs, ep_ret, steps = next_obs, ep_ret + rew, steps + 1
            done = term or trunc
        if done:
            returns.append(ep_ret)
        trajs += 1
        if use_posterior:
            if trajs % update_posterior_every == 0 and replay.enc[task].size > 0:
                ctx = replay.sample_context([task], context_batch_size, device)
                agent.infer_posterior(agent.make_context(ctx["obs"], ctx["act"], ctx["rew"], ctx["next_obs"]))
            else:
                agent.sample_z()
        else:
            agent.clear_z()
        if max_trajectories is not None and trajs >= max_trajectories:
            break
    agent.train()
    return {"steps": steps, "trajectories": trajs, "mean_return": float(np.mean(returns)) if returns else float("nan")}
