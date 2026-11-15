# PEARL-Torch: Off-Policy Meta-RL with Probabilistic Context Variables

A PyTorch implementation of **PEARL** (Rakelly et al., 2019, *Efficient Off-Policy Meta-Reinforcement
Learning via Probabilistic Context Variables*, [arXiv:1903.08254](https://arxiv.org/abs/1903.08254)),
laid out like [RL2-Torch-Experiments](../RL2-Torch-Experiments-main) and using the **same three
environments**, so the two context-based meta-RL methods can be compared directly:

| Experiment | Env | Task parameter | Actions |
|---|---|---|---|
| `experiments/cartpole_wind.py` | CartPole + wind force, reward `cos(pole angle)` | wind ∈ [-5, 5] | discrete (2) |
| `experiments/pendulum_wind.py` | Pendulum + wind torque | torque ∈ [-0.5, 0.5] | continuous, bounds ±2 |
| `experiments/halfcheetah_action.py` | HalfCheetah-v5 with additive action offset | offset ∈ [-0.2, 0.2] | continuous, bounds ±1 (6-D) |

The environment code (`CartPoleEnvWithWind`, `RewardShapingWrapper`, `sample_tasks_*`, `make_envs`, …) is copied unchanged from the RL² repo.

## How PEARL differs from RL²

|  | RL² (PPO) | PEARL (SAC) |
|---|---|---|
| Task belief | GRU hidden state `h_t` | latent `z ~ q(z│c)`, product of per-transition Gaussians |
| Context | the whole history, ordered | unordered set of transitions `(s,a,r,s')` |
| Learning | on-policy PPO | off-policy SAC, per-task replay buffers |
| Exploration at test time | implicit in the RNN | posterior sampling: sample `z`, act for one trajectory, update posterior |

```
Meta-training (Alg. 1)                         Meta-testing (Alg. 2)
 for each task k in a fixed train set           z ~ N(0, I)
   z ~ prior       -> collect -> RL + enc buf    for trajectory 1..K:
   z ~ q(z|c_enc)  -> collect -> RL + enc buf       run pi(a|s,z); add to context c
   z ~ q(z|c_enc)  -> collect -> RL buf only        z ~ q(z|c)           # posterior sampling
 for each gradient step:
   context batch c  <- encoder buffer (recent)
   RL batch         <- RL buffer (everything)
   z ~ q(z|c);  critic loss (grad into encoder) + beta*KL;  V loss, actor loss with z detached
```

## Structure

```
agents/
  nn_tools.py        orthogonal-init MLP (same as RL²)
  pearl_agent.py     ContextEncoder (product of Gaussians), PEARLAgent: policy, twin Q, V, target V
utils/
  buffer.py          per-task RL buffer + encoder (context) buffer
  rollout.py         SingleTaskEnv, collect_steps (prior / posterior data collection)
  evaluation.py      run_meta_eval: posterior-sampling evaluation, returns per trajectory
  distributions.py   TanhNormal (same as RL²)     normalizers.py  (same as RL²)
train.py             train_pearl(...) + pearl_update(...)   (counterpart of train_rl2_ppo)
experiments/         one script per environment
tests/               unit tests (pytest)
```

`train_pearl` takes the same factories as `train_rl2_ppo` (`model_factory`, `sample_tasks_train/eval`,
`make_envs_train/eval`, `action_dim`, `is_discrete`, `action_low/high`, `save_path`); every field of
`PEARLConfig` can be overridden by keyword.

## Install and run (uv)

This project uses [uv](https://docs.astral.sh/uv/) ([install uv](https://docs.astral.sh/uv/getting-started/installation/)). From the repo root:

```bash
uv sync                          # creates .venv, installs torch, numpy, gymnasium and pytest
uv sync --extra mujoco           # additionally installs mujoco (needed for HalfCheetah)
```

Train (one script per environment; the best policy is written to `checkpoints/`):

```bash
uv run python experiments/cartpole_wind.py
uv run python experiments/pendulum_wind.py
uv run python experiments/halfcheetah_action.py      # needs the mujoco extra
```

Run the unit tests:

```bash
uv run pytest
```

Long runs on a server (keeps running after you disconnect):

```bash
nohup uv run python experiments/cartpole_wind.py > cartpole.log 2>&1 &
```

On a CPU-only machine, torch's default wheel may pull in large CUDA libraries. To use the CPU build, run
`uv pip install torch --index-url https://download.pytorch.org/whl/cpu` after `uv sync`, and start scripts with `uv run --no-sync`.

During training the log prints the mean return of held-out tasks for each evaluation trajectory
(`traj 1` uses the prior; `traj k` uses the posterior after `k-1` trajectories). A working
meta-learner shows returns increasing along the trajectory index.

`checkpoints/<name>.pt` always holds the best policy so far: it is overwritten only when the mean return on the last evaluation trajectory improves (selected on the same eval tasks it is scored on, so that number is optimistically biased).

Load a checkpoint (run from the repo root, e.g. with `uv run python`):

```python
from experiments.cartpole_wind import cartpole_model_factory
from train import load_checkpoint
agent = load_checkpoint(cartpole_model_factory(), "checkpoints/cartpole_wind.pt")
```

## Implementation notes and deviations (please read)

* **Discrete SAC for CartPole.** The paper's SAC is continuous. For CartPole the policy is
  categorical, `Q(s,z)` outputs one value per action, and the V target uses the exact expectation
  over actions. For continuous envs the policy is a tanh-Gaussian (reusing `TanhNormal`), actions are stored in [-1, 1] and mapped to the env bounds.
* **Fixed entropy weight 1 with reward scaling** (`reward_scale`), as in the authors' setup, instead of a learned temperature.
* **Termination vs. time limit.** The buffer stores `terminated` only, so `V_bar` bootstraps at `TimeLimit` truncations.
* **One trajectory = one episode** (500 / 200 / 1000 steps for CartPole / Pendulum / HalfCheetah), unlike RL²'s 1024-step trial of consecutive episodes. Meta-testing therefore uses a few consecutive episodes on the same task.
* **Observation normalisation** with the RL² `RunningMeanStd1DNormalizer` (frozen after `max_samples`); rewards are normalised only inside the context, the Bellman target uses the raw reward × `reward_scale`.
* **Hyper-parameters.** The paper PDF has no hyper-parameter table. Defaults marked *(authors)* in `PEARLConfig`
  (latent dim 5, meta-batch 16, batch 256, context batch 100, lr 3e-4, τ 0.005, reward scale 5, β 0.1 with the KL summed over tasks, policy regularisers 1e-3) are
  from my recollection of the authors' public code and were **not** verified against it. The network widths and the number of training tasks/iterations in `experiments/*.py` are my own choices, sized for CPU runs. Tune them per environment.
* The deterministic-context and RNN-encoder ablations of the paper are not included.

## Status

* Unit tests pass (`pytest tests`, 11 tests); all three experiment scripts run end to end.
* One short CartPole-wind run (40 iterations, 20 training tasks, 8 evenly spaced eval tasks, seed 0, about 26 min on 2 CPU cores) took the
  mean eval return on the last trajectory from about 10 (untrained) to about 480 of a maximum near 500. Evaluation
  at iterations 35 and 40 gave 478.6 and 477.2 (3rd trajectory), with 343.6 / 352.7 on the prior-`z` first trajectory. This is one seed and a small eval set, so treat it as a sanity check, not a benchmark.
* Pendulum and HalfCheetah were only smoke-tested (a few updates), not trained.


## Acknowledgements

- Implementation developed with assistance from Claude (Anthropic, Claude Sonnet 5.5), which wrote the initial code; the author reviewed it and ran the experiments.
- Algorithm: Rakelly et al., "Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables", ICML 2019 ([arXiv:1903.08254](https://arxiv.org/abs/1903.08254)).
- Environments, `TanhNormal`, normalizers and MLP helper are taken from RL2-Torch-Experiments.