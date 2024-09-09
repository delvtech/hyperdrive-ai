import os

import ray
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig

from traiderdaive.ray_environments.ray_hyperdrive_env import AGENT_PREFIX, POLICY_PREFIX, RayHyperdriveEnv

GPU = False

env_config = RayHyperdriveEnv.Config(eval_mode=True)
policies = [POLICY_PREFIX + str(i) for i in range(env_config.num_agents)]

ray.init(local_mode=True)  # Use local_mode=True for debugging
config: AlgorithmConfig = (
    PPOConfig()
    .environment(env=RayHyperdriveEnv, env_config={"env_config": env_config})
    .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
    .multi_agent(
        policies=set(policies),
        # Mapping agent0 to policy0, etc.
        policy_mapping_fn=(lambda agent_id, episode, **kwargs: f"{POLICY_PREFIX}{agent_id.lstrip(AGENT_PREFIX)}"),
        policies_to_train=policies,
    )
    .evaluation(evaluation_duration=1, evaluation_duration_unit="episodes")
)
algo = config.build()
run_timestamp = "2024_09_09_00_00_00"
idx = "000001"
checkpoint_dir = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/checkpoints/{run_timestamp}/{idx}"
algo.restore(checkpoint_dir)
# env = algo.workers.local_env_runner.env.env
algo.evaluate()
print("Finished evaluation")
