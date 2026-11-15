"""Per-task replay buffers for PEARL.

PEARL keeps, for every training task, two buffers (Sec. 4.3, Fig. 3 of the paper):

* an **RL buffer** with all data ever collected for that task (off-policy SAC batches);
* an **encoder (context) buffer** holding only recently collected data, close to the
  current policy, from which context batches for ``q_phi(z|c)`` are drawn.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch


class TransitionBuffer:
    """Ring buffer of ``(s, a, r, s', done)`` for one task.

    Parameters
    ----------
    capacity : int
    obs_dim : int
    action_dim : int
        Action vector size (continuous) or ignored for storage (discrete).
    is_discrete : bool
        Discrete actions are stored as int64 scalars, continuous as float32 vectors.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, is_discrete: bool):
        self.capacity, self.is_discrete = capacity, is_discrete
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity,), np.int64) if is_discrete else np.zeros((capacity, action_dim), np.float32)
        self.rew = np.zeros((capacity, 1), np.float32)
        self.done = np.zeros((capacity, 1), np.float32)  # true termination only, NOT time-limit truncation
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done) -> None:
        """Append one transition."""
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.next_obs[i], self.done[i] = next_obs, float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def clear(self) -> None:
        self.ptr = 0
        self.size = 0

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Uniform sample with replacement. Each value has leading dim ``batch_size``."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return {k: getattr(self, k)[idx] for k in ("obs", "act", "rew", "next_obs", "done")}


class MultiTaskReplay:
    """RL + encoder buffers for ``num_tasks`` training tasks.

    Parameters
    ----------
    num_tasks : int
    obs_dim, action_dim : int
    is_discrete : bool
    rl_capacity : int
        Capacity of each task's RL buffer.
    enc_capacity : int
        Capacity of each task's encoder buffer.
    """

    def __init__(self, num_tasks: int, obs_dim: int, action_dim: int, is_discrete: bool,
                 rl_capacity: int = 1_000_000, enc_capacity: int = 10_000):
        self.num_tasks = num_tasks
        self.rl = [TransitionBuffer(rl_capacity, obs_dim, action_dim, is_discrete) for _ in range(num_tasks)]
        self.enc = [TransitionBuffer(enc_capacity, obs_dim, action_dim, is_discrete) for _ in range(num_tasks)]

    def add(self, task: int, obs, act, rew, next_obs, done, to_enc: bool = True) -> None:
        """Add one transition to the RL buffer (always) and the encoder buffer (if ``to_enc``)."""
        self.rl[task].add(obs, act, rew, next_obs, done)
        if to_enc:
            self.enc[task].add(obs, act, rew, next_obs, done)

    def clear_enc(self, task: int) -> None:
        """Empty one task's encoder buffer (done before re-collecting that task)."""
        self.enc[task].clear()

    def _stack(self, bufs: List[TransitionBuffer], tasks: Sequence[int], batch_size: int,
               device: torch.device) -> Dict[str, torch.Tensor]:
        parts = [bufs[t].sample(batch_size) for t in tasks]
        return {k: torch.as_tensor(np.stack([p[k] for p in parts]), device=device) for k in parts[0]}

    def sample_rl(self, tasks: Sequence[int], batch_size: int, device) -> Dict[str, torch.Tensor]:
        """RL batch. Values have shape (T, B, ...)."""
        return self._stack(self.rl, tasks, batch_size, device)

    def sample_context(self, tasks: Sequence[int], batch_size: int, device) -> Dict[str, torch.Tensor]:
        """Context batch from the encoder buffers. Values have shape (T, N, ...)."""
        return self._stack(self.enc, tasks, batch_size, device)
