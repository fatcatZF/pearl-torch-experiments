"""Meta-test evaluation by posterior sampling (PEARL Algorithm 2)."""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import torch

from utils.rollout import SingleTaskEnv


@torch.no_grad()
def run_meta_eval(
    agent,
    make_envs: Callable,
    tasks: List[dict],
    device,
    is_discrete: bool,
    action_low=None,
    action_high=None,
    num_trajectories: int = 3,
    deterministic_last: bool = True,
) -> Dict[str, np.ndarray]:
    """Evaluate adaptation on held-out tasks.

    For every task: start from the prior ``z ~ N(0, I)``; run ``num_trajectories``
    consecutive episodes; after each episode the posterior ``q(z | c)`` is recomputed
    from **all** transitions gathered on this task so far, and a new ``z`` is sampled
    (posterior sampling, Alg. 2).  The last trajectory is the "adapted" performance
    and uses deterministic actions if ``deterministic_last``.

    Parameters
    ----------
    agent : PEARLAgent
    make_envs : callable(list[dict]) -> SyncVectorEnv
    tasks : list of task dicts (one env per task)
    num_trajectories : int
        Trajectory 0 uses the prior, trajectory k>0 the posterior after k episodes.

    Returns
    -------
    dict
        ``returns``: array (num_tasks, num_trajectories) of undiscounted returns;
        ``mean``: (num_trajectories,) mean over tasks.
    """
    was_training = agent.training
    agent.eval()
    out = np.zeros((len(tasks), num_trajectories))
    for i, task in enumerate(tasks):
        env = SingleTaskEnv(make_envs([task]), is_discrete, action_low, action_high)
        agent.clear_z()
        ctx_buf: List[torch.Tensor] = []
        for k in range(num_trajectories):
            det = deterministic_last and k == num_trajectories - 1
            obs, done, ret = env.reset(), False, 0.0
            traj = {"o": [], "a": [], "r": [], "o2": []}
            while not done:
                a = agent.act(torch.as_tensor(obs, device=device), deterministic=det)
                nxt, r, term, trunc = env.step(a)
                traj["o"].append(obs); traj["a"].append(a); traj["r"].append([r]); traj["o2"].append(nxt)
                obs, ret, done = nxt, ret + r, term or trunc
            out[i, k] = ret
            t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=device)
            ctx_buf.append(agent.make_context(t(traj["o"]), t(traj["a"], torch.int64 if is_discrete else torch.float32),
                                              t(traj["r"]), t(traj["o2"])))
            agent.infer_posterior(torch.cat(ctx_buf, dim=0))
        env.close()
    agent.train(was_training)
    return {"returns": out, "mean": out.mean(0)}
