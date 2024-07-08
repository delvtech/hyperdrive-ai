"""Simple stable baselines policy trainer"""

import os

import gymnasium as gym
import ray

from ray.rllib.algorithms import ppo
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls

# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.monitor import Monitor, load_results
# from stable_baselines3.common.results_plotter import ts2xy

# Import registers hyperdrive envs
# from agent0.traiderdaive import FullHyperdriveEnv
from gym_environments.ray_hyperdrive_env import RayHyperdriveEnv

# Policy params listed in /ray/rllib/models/catalog.py
model_params = {"uses_new_env_runners": True, "vf_share_layers": False}
# PPO params listed in /ray/rllib/algorithms/ppo/ppo.py
ppo_params = {
    # MDP discount factor TODO: Is this used in PPO?
    "gamma": 0.99,
    # Scheduler: [[timestep, lr-value], [timestep, lr-value], ...]
    "lr_schedule": None,
    # Learning rate
    "lr": 5e-5,
    # Batch size per worker: total batch size = `num_learners` x `train_batch_size_per_learner`
    "train_batch_size_per_learner": None,
    # Use critic as baseline (Required for using GAE)
    "use_critic": True,
    # Use Generalized Advantage Estimator
    "use_gae": True,
    # GAE lambda parameter
    "lambda_": 1.0,
    # Use KL term in loss function
    "use_kl_loss": True,
    # Initial coefficient for KL divergence
    "kl_coeff": 0.2,
    # Target value for KL divergence
    "kl_target": 0.01,
    # Mini batch of train batch
    "mini_batch_size_per_learner": None,
    # Number of update iterations per train batch (num epochs)
    "num_sgd_iter": 30,
    # Shuffle sequences in the train batch
    "shuffle_sequences": True,
    # Value function loss coefficient
    "vf_loss_coeff": 1.0,
    # Coefficient for entropy regularizer
    "entropy_coeff": 0.0,
    # Decay schedule for the entropy regularizer
    "entropy_coeff_schedule": None,
    # PPO clip parameter
    "clip_param": 0.3,
    # Clip parameter for the value function. (Sensitive to scale of reward; if V is large, increase this)
    "vf_clip_param": 10.0,
    # Clip Gradients
    "grad_clip": None,
    "grad_clip_by": "global_norm",
    "model": model_params,
}


def run_train():
    """Runs training to generate a RL model."""
    # TODO parameterize these variables
    # gym_config = RayHyperdriveEnv.Config()
    # env = gym.make(id="traiderdaive/full_hyperdrive_env", gym_config=gym_config)
    # env.chain.run_dashboard()
    # env = Monitor(env, log_dir)

    # TODO: Make all of init() params explicit
    ray.init(local_mode=True)  # Use local_mode=True for debugging
    config = (
        ppo.PPOConfig()
        # .environment(env=RayHyperdriveEnv, env_config=asdict(gym_config))
        .environment(env=RayHyperdriveEnv)
        .api_stack(
            enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True
        )
        .env_runners(num_env_runners=2)
        .resources(num_cpus_for_main_process=1)
        .learners(num_learners=0, num_gpus_per_learner=0)
        .training(**ppo_params)
        .multi_agent(
            policies={"policy0", "policy1"},
            # Simple mapping fn, mapping agent0 to main0 and agent1 to main1.
            policy_mapping_fn=(
                lambda agent_id, episode, **kwargs: f"policy{agent_id[-1]}"
            ),
            policies_to_train=["policy0", "policy1"],
        )
    )
    algo = config.build()
    # algo.train()
    ray.shutdown()


if __name__ == "__main__":
    run_train()
