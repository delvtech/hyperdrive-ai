"""Multi agent rllib policy trainer for burrito attack"""

import os
import shutil
from datetime import datetime

import ray
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

from traiderdaive.ray_environments.ray_hyperdrive_env import AGENT_PREFIX, POLICY_PREFIX
from traiderdaive.ray_environments.attack_hyperdrive_env import AttackHyperdriveEnv
from traiderdaive.ray_environments.rewards import TotalRealizedValue
from traiderdaive.ray_environments.variable_rate_policy import ConstantVariableRate

GPU = True


def run_train():
    """Runs training to generate a RL model."""
    # TODO parameterize these variables
    # TODO Does the env need to be registered with ray?
    # TODO Why does Ray add an env runner to num_env_runners? How is this used?
    # TODO Setup monitoring and saving checkpoints
    # TODO Use ray train/tune?
    # TODO Make sure env config and algo config are saved somehow for reproducibility
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%I:%M:%S %p')}")

    # Reward type
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
    )
    policies = [POLICY_PREFIX + str(i) for i in range(env_config.num_agents)]

    # TODO: Make all of init() params explicit
    ray.init(local_mode=False, _temp_dir=os.path.expanduser("~/ray_results/tmp"))  # Use local_mode=True for debugging
    config: AlgorithmConfig = (
        PPOConfig()
        .environment(env=AttackHyperdriveEnv, env_config={"env_config": env_config})
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        .env_runners(
            num_env_runners=5,
            num_envs_per_env_runner=1,
            sample_timeout_s=120,
            rollout_fragment_length="auto",
        )
        .resources(num_cpus_for_main_process=1, num_gpus=1 if GPU else 0)
        .learners(
            num_learners=0,
            num_gpus_per_learner=1 if GPU else 0,
            num_cpus_per_learner=0 if GPU else 1,
        )
        .multi_agent(
            policies=set(policies),
            # Mapping agent0 to policy0, etc.
            policy_mapping_fn=(lambda agent_id, episode, **kwargs: f"{POLICY_PREFIX}{agent_id.lstrip(AGENT_PREFIX)}"),
            policies_to_train=policies,
        )
        .training(
            # PPO params listed in /ray/rllib/algorithms/ppo/ppo.py
            # MDP discount factor TODO: Is this used in PPO?
            gamma=0.99,
            # Scheduler: [[timestep, lr-value], [timestep, lr-value], ...]
            lr_schedule=None,
            # Learning rate
            lr=5e-5,
            # Batch size per worker=total batch size = `num_learners` x `train_batch_size_per_learner`
            train_batch_size_per_learner=env_config.num_episodes_per_update * env_config.episode_length,
            # Mini batch of train batch
            mini_batch_size_per_learner=env_config.num_episodes_per_update * env_config.episode_length,
            # Number of update iterations per train batch (num epochs)
            num_sgd_iter=env_config.num_epochs_sgd,
            # Use critic as baseline (Required for using GAE)
            use_critic=True,
            # Use Generalized Advantage Estimator
            use_gae=True,
            # GAE lambda parameter
            lambda_=0.95,
            # Use KL term in loss function
            use_kl_loss=True,
            # Initial coefficient for KL divergence
            kl_coeff=0.2,
            # Target value for KL divergence
            kl_target=0.01,
            # Shuffle sequences in the train batch
            shuffle_sequences=True,
            # Value function loss coefficient
            vf_loss_coeff=1.0,
            # Coefficient for entropy regularizer
            entropy_coeff=0.0,
            # Decay schedule for the entropy regularizer
            entropy_coeff_schedule=None,
            # PPO clip parameter
            clip_param=0.3,
            # Clip parameter for the value function. (Sensitive to scale of reward; if V is large, increase this)
            vf_clip_param=10.0,
            # Clip Gradients
            grad_clip=None,
            grad_clip_by="global_norm",
            # Model params
            # TODO: vf_share_layers: Should actor and critic share layers? https://arxiv.org/abs/2006.05990
            model={"uses_new_env_runners": True, "vf_share_layers": False},
        )
    )
    algo = config.build()
    run_timestamp = datetime.now().strftime("BurritoAttack_%Y_%m_%d_%H_%M_%S")
    for i in range(env_config.num_training_loops):
        timestamp = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
        print(f"\n-----------------\nTraining Iteration {i}\nTime: {timestamp}")
        result = algo.train()
        print(pretty_print(result))
        project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        checkpoint_dir = f"{project_dir}/checkpoints/{run_timestamp}/{i:06d}"
        save_result = algo.save(checkpoint_dir=checkpoint_dir)
        print(f"Saved checkpoint to: {save_result.checkpoint.path}")
        # Remove tmp files created by anvil
        tmp_dir = os.path.expanduser("~/.foundry/anvil/tmp")
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
    end_time = datetime.now()
    print(f"\nFinished: {end_time.strftime('%I:%M:%S %p')}")
    print(f"({(end_time - start_time).total_seconds() / 60} minutes.)")
    ray.shutdown()


if __name__ == "__main__":
    run_train()