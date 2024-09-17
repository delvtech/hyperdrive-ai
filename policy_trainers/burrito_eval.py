# %%
import os

import numpy as np
import ray
import torch
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig

from traiderdaive.ray_environments.attack_hyperdrive_env import AttackHyperdriveEnv
from traiderdaive.ray_environments.ray_hyperdrive_env import AGENT_PREFIX, POLICY_PREFIX
from traiderdaive.ray_environments.rewards import TotalRealizedValue
from traiderdaive.ray_environments.variable_rate_policy import ConstantVariableRate

GPU = False

reward = TotalRealizedValue

# Rate policy
rate_policy = ConstantVariableRate()

env_config = AttackHyperdriveEnv.Config(
    variable_rate_policy=rate_policy,
    reward_policy=reward,
    num_random_bots=0,
    num_random_hold_bots=0,
    num_agents=1,
    episode_length=50,
    step_advance_time=24 * 3600,  # 24 hrs
    max_positions_per_type=1,
    eval_mode=True,
)
policy_ids = [POLICY_PREFIX + str(i) for i in range(env_config.num_agents)]
agent_ids = [AGENT_PREFIX + str(i) for i in range(env_config.num_agents)]

ray.init(local_mode=True)  # Use local_mode=True for debugging
config: AlgorithmConfig = (
    PPOConfig()
    .environment(env=AttackHyperdriveEnv, env_config={"env_config": env_config})
    .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
    .multi_agent(
        policies=set(policy_ids),
        # Mapping agent0 to policy0, etc.
        policy_mapping_fn=(lambda agent_id, episode, **kwargs: f"{POLICY_PREFIX}{agent_id.lstrip(AGENT_PREFIX)}"),
        policies_to_train=policy_ids,
    )
    .evaluation(evaluation_duration=1, evaluation_duration_unit="episodes")
)
algo = config.build()
# run_timestamp = "2024_09_09_00_00_00"
# idx = "000000"
run_timestamp = "BurritoAttack_Continued_2024_09_15_15_32_52"
idx = "000978"


def run_one_step(env_runner):
    """Runs a single step in the environment.
    Args:
        env_runner (MultiAgentEnvRunner): algo.env_runner
    Returns:
        step sample
    """
    sample = env_runner.sample(num_timesteps=1, explore=False)
    return sample


def get_actions(module, observations):
    """Raw actions from RL module."""
    to_module = {}
    for agent_id, policy_id in zip(agent_ids, policy_ids):
        # module = algo.get_module(policy_id)
        # policy_modules[policy_id] = module
        # obs_dict = {"obs": torch.from_numpy(np.array([observations[agent_id]]))}
        to_module[policy_id] = {"obs": torch.from_numpy(np.array([observations[agent_id]]))}

    logits_dict = module.forward_inference(to_module)
    action_dict = {}
    for agent_id, policy_id in zip(agent_ids, policy_ids):
        action_logits = logits_dict[policy_id]["action_dist_inputs"]
        action = torch.argmax(action_logits[0]).numpy()
        action_dict[agent_id] = action


# %%
checkpoint_dir = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/checkpoints/{run_timestamp}/{idx}"
algo.restore(checkpoint_dir)

# Run custom number of steps:
# for step in range(env_config.episode_length):
#     sample = run_one_step(algo.env_runner)


# Run full eval:
algo.evaluate()  # Run full eval
print("Finished evaluation")

chain = algo.env_runner.env.env.chain
pool = chain._deployed_hyperdrive_pools[0]

# TODO: Expand on this to make manual action -> step() -> observation possible
# Get environment and RL module
# env = algo.env_runner.env.env
# module = algo.env_runner.module
# Get observations and make sure we're at the beginning of an episode
# observations, info = env.reset()
# obs, reward, terminated, truncated, info = env.step(action_dict)

# %%