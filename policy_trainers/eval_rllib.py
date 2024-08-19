# %%

import ray
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
import pathlib

from traiderdaive.ray_environments.ray_hyperdrive_env import AGENT_PREFIX, POLICY_PREFIX, RayHyperdriveEnv

GPU = False

env_config = RayHyperdriveEnv.Config(eval_mode=True)
policies = [POLICY_PREFIX + str(i) for i in range(env_config.num_agents)]

# TODO: Make all of init() params explicit
ray.init(local_mode=True)  # Use local_mode=True for debugging
config: AlgorithmConfig = (
    PPOConfig()
    .environment(env=RayHyperdriveEnv, env_config={"env_config": env_config})
    .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
    # .env_runners(
    #     num_env_runners=0,
    #     num_envs_per_env_runner=1,
    #     sample_timeout_s=120,
    #     rollout_fragment_length="auto",
    # )
    # .resources(num_cpus_for_main_process=1, num_gpus=1 if GPU else 0)
    .multi_agent(
        policies=set(policies),
        # Mapping agent0 to policy0, etc.
        policy_mapping_fn=(lambda agent_id, episode, **kwargs: f"{POLICY_PREFIX}{agent_id.lstrip(AGENT_PREFIX)}"),
        policies_to_train=policies,
    )
    .evaluation(evaluation_duration=1, evaluation_duration_unit="episodes")
)
algo = config.build()
checkpoint_dir = "/Users/wshainin/workspace/DELV/hyperdrive-ai/checkpoints/"
algo.restore(checkpoint_dir)
# algo = Algorithm.from_checkpoint(checkpoint_dir)
algo.evaluate()
print("DONE")
# %%
