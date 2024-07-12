"""Multi agent rllib policy trainer"""

import os

import ray

from ray.rllib.algorithms import ppo
from ray.tune.logger import pretty_print

from traiderdaive.ray_environments.ray_hyperdrive_env import POLICY_PREFIX, RayHyperdriveEnv

# Policy params listed in /ray/rllib/models/catalog.py
# TODO: vf_share_layers: Should actor and critic share layers? https://arxiv.org/abs/2006.05990
model_params = {"uses_new_env_runners": True, "vf_share_layers": False}
# PPO params listed in /ray/rllib/algorithms/ppo/ppo.py
# TODO: Should this be in RayHyperdriveEnv.Config?
ppo_params = {
    # MDP discount factor TODO: Is this used in PPO?
    "gamma": 0.99,
    # Scheduler: [[timestep, lr-value], [timestep, lr-value], ...]
    "lr_schedule": None,
    # Learning rate
    "lr": 5e-5,
    # Batch size per worker: total batch size = `num_learners` x `train_batch_size_per_learner`
    "train_batch_size_per_learner": 10,
    # Mini batch of train batch
    "mini_batch_size_per_learner": 10,
    # Number of update iterations per train batch (num epochs)
    "num_sgd_iter": 10,
    # Use critic as baseline (Required for using GAE)
    "use_critic": True,
    # Use Generalized Advantage Estimator
    "use_gae": True,
    # GAE lambda parameter
    "lambda_": 0.95,
    # Use KL term in loss function
    "use_kl_loss": True,
    # Initial coefficient for KL divergence
    "kl_coeff": 0.2,
    # Target value for KL divergence
    "kl_target": 0.01,
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
    # TODO Does the env need to be registered with ray?
    # TODO Why does Ray add an env runner to num_env_runners? How is this used?
    # TODO Setup monitoring and saving checkpoints
    gym_config = RayHyperdriveEnv.Config()
    # env.chain.run_dashboard()
    policies = [POLICY_PREFIX + str(i) for i in range(gym_config.num_agents)]

    # TODO: Make all of init() params explicit
    ray.init(local_mode=True)  # Use local_mode=True for debugging
    config = (
        ppo.PPOConfig()
        # .environment(env=RayHyperdriveEnv, env_config=asdict(gym_config))
        # TODO: Not sure about the best way to pass config to env
        .environment(env=RayHyperdriveEnv, env_config={"gym_config": gym_config})
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        .env_runners(num_env_runners=2, num_envs_per_env_runner=1)
        .resources(num_cpus_for_main_process=1)
        .learners(num_learners=0, num_gpus_per_learner=0, num_cpus_per_learner=1)
        .training(**ppo_params)
        .multi_agent(
            policies=set(policies),
            # Simple mapping fn, mapping agent0 to policy0
            policy_mapping_fn=(lambda agent_id, episode, **kwargs: f"{POLICY_PREFIX}{agent_id[-1]}"),
            policies_to_train=policies,
        )
    )
    algo = config.build()
    for i in range(2):
        print(f"Training iteration {i}...")
        result = algo.train()
        print(pretty_print(result))

    ray.shutdown()


if __name__ == "__main__":
    run_train()
