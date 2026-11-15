import gymnasium as gym 
from gymnasium.wrappers import TransformReward, TransformAction, TransformObservation
from gymnasium.vector import SyncVectorEnv
from gymnasium.spaces import Box


import numpy as np 


from agents.pearl_agent import PEARLAgent
from utils.normalizers import RunningMeanStd1DNormalizer


from train import train_pearl

import os 
import random

from typing import List, Dict, Any 








def sample_tasks_train(num_envs: int = 4) -> List[Dict[str, Any]]:
    """
    sample action modifier
    """
    tasks = []
    for _ in range(num_envs):
        #sample action modifier
        a_m = np.random.uniform(low=-0.2, high=0.2) 
        tasks.append({
            "a_m": a_m
        })
    return tasks




def sample_tasks_eval(num_envs: int = 16) -> List[Dict[str, Any]]:
    """
    sample action modifier
    """
    a_ms = np.linspace(-0.2, 0.2, num=num_envs)
    
    tasks = []
    for a_m in a_ms:
        tasks.append({
            "a_m": float(a_m)
        })
    return tasks




def make_envs(task_configs: List[Dict[str, Any]]):
    def make_env(config):
        a_m = config["a_m"]
        return lambda: TransformAction(gym.make("HalfCheetah-v5"),
                                                lambda a: a+a_m,
                                                Box(-1.0, 1.0, shape=(6,), dtype=np.float32))
    return SyncVectorEnv([make_env(cfg) for cfg in task_configs])




def halfcheetah_model_factory(latent_dim=5, hidden_dims=(256, 256, 256)):
    """
    Create the PEARL agent for this environment (obs_dim=17, action_dim=6, continuous).
    """
    obs_dim, action_dim = 17, 6
    obs_norm = RunningMeanStd1DNormalizer(shape=(obs_dim,), max_samples=20000)
    rew_norm = RunningMeanStd1DNormalizer(shape=(1,), max_samples=20000)
    return PEARLAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        is_discrete=False,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        obs_norm=obs_norm,
        rew_norm=rew_norm,
    )




if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    save_path = os.path.join(project_root, "checkpoints", "halfcheetah.pt")

    trained_agent = train_pearl(
        model_factory=halfcheetah_model_factory,
        sample_tasks_train=sample_tasks_train,
        make_envs_train=make_envs,
        sample_tasks_eval=sample_tasks_eval,
        make_envs_eval=make_envs,
        action_dim=6,
        is_discrete=False,
        action_low=-1.0,
        action_high=1.0,
        save_path=save_path,
        iterations=500,
        num_train_tasks=20,
        steps_initial=2000,
        steps_prior=1000,
        steps_posterior=1000,
        steps_extra_rl=1000,
        train_steps_per_iter=1000,
        reward_scale=5.0,
        eval_every=20,
    )
