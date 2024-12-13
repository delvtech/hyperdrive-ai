# %% ###### Setup ######

import os

import numpy as np
import ray
import torch
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig

from traiderdaive.ray_environments.attack_hyperdrive_env import AttackHyperdriveEnv
from traiderdaive.ray_environments.hyperdrive_env import AGENT_PREFIX, POLICY_PREFIX
from traiderdaive.ray_environments.rewards import TotalRealizedValue
from traiderdaive.ray_environments.variable_rate_policy import ConstantVariableRate

GPU = False

reward = TotalRealizedValue

# Rate policy
rate_policy = ConstantVariableRate()

env_config = AttackHyperdriveEnv.Config(
    variable_rate_policy=rate_policy,
    reward_policy=reward,
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
idx = "002740"


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


###### Run eval ######
checkpoint_dir = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/checkpoints/{run_timestamp}/{idx}"
algo.restore(checkpoint_dir)

# Run custom number of steps:
# for step in range(env_config.episode_length):
#     sample = run_one_step(algo.env_runner)


# Run full eval:
algo.evaluate()  # Run full eval
print("Finished evaluation")

# %% ###### Analysis ######
###### Get underlying agent0 objects ######
from agent0 import LocalChain, LocalHyperdrive
from agent0.core.hyperdrive.interactive.local_hyperdrive_agent import LocalHyperdriveAgent

chain: LocalChain = algo.env_runner.env.env.chain
pool: LocalHyperdrive = chain._deployed_hyperdrive_pools[0]
agent: LocalHyperdriveAgent = chain._chain_agents[0]

# TODO: Expand on this to make manual action -> step() -> observation possible
# Get environment and RL module
# env = algo.env_runner.env.env
# module = algo.env_runner.module
# Get observations and make sure we're at the beginning of an episode
# observations, info = env.reset()
# obs, reward, terminated, truncated, info = env.step(action_dict)

# %% ###### Get and prepare trades and positions from db ######
from fixedpointmath import FixedPoint

pool_info = pool.get_pool_info(coerce_float=True)

# Get mapping from block to timestamp
timestamp_lookup = pool_info[
    [
        "block_number",
        "timestamp",
        "epoch_timestamp",
    ]
]

# Modify and get relevant columns of dataframes
trade_events = pool.get_trade_events(coerce_float=True).merge(right=timestamp_lookup, on="block_number")[
    [
        "block_number",
        "timestamp",
        "username",
        "event_type",
        "token_type",
        "token_id",
        "token_delta",
        "base_delta",
        "epoch_timestamp",
    ]
]

positions = pool.get_historical_positions(coerce_float=True).merge(right=timestamp_lookup, on="block_number")[
    [
        "block_number",
        "timestamp",
        "username",
        "token_type",
        "token_id",
        "token_balance",
        "unrealized_value",
        "realized_value",
        "pnl",
        "epoch_timestamp",
    ]
]

pnl = pool.get_historical_pnl(coerce_float=True).merge(right=timestamp_lookup, on="block_number")[
    ["block_number", "timestamp", "username", "pnl", "epoch_timestamp"]
]

# Add normalized pnl
pnl["budget"] = np.nan
pnl.loc[pnl["username"] != "agent0", "budget"] = float(FixedPoint(100_000))
pnl.loc[pnl["username"] == "agent0", "budget"] = float(env_config.rl_agent_budget)

pnl["normalized_pnl"] = pnl["pnl"] / pnl["budget"]


# %% ###### Get and prepare final positions of bad agent ######
agent_positions = agent.get_positions(show_closed_positions=True, coerce_float=True)
agent_positions = agent_positions[
    ["block_number", "token_type", "token_id", "token_balance", "unrealized_value", "realized_value", "pnl"]
]

# Combine withdrawal shares into LP
agent_positions.loc[agent_positions["token_id"] == "LP", "token_balance"] = (
    agent_positions.loc[agent_positions["token_id"] == "LP", "token_balance"].values[0]
    + agent_positions.loc[agent_positions["token_id"] == "WITHDRAWAL_SHARE", "token_balance"].values[0]
)

agent_positions.loc[agent_positions["token_id"] == "LP", "unrealized_value"] = (
    agent_positions.loc[agent_positions["token_id"] == "LP", "unrealized_value"].values[0]
    + agent_positions.loc[agent_positions["token_id"] == "WITHDRAWAL_SHARE", "unrealized_value"].values[0]
)

agent_positions.loc[agent_positions["token_id"] == "LP", "realized_value"] = (
    agent_positions.loc[agent_positions["token_id"] == "LP", "realized_value"].values[0]
    + agent_positions.loc[agent_positions["token_id"] == "WITHDRAWAL_SHARE", "realized_value"].values[0]
)

agent_positions.loc[agent_positions["token_id"] == "LP", "pnl"] = (
    agent_positions.loc[agent_positions["token_id"] == "LP", "pnl"].values[0]
    + agent_positions.loc[agent_positions["token_id"] == "WITHDRAWAL_SHARE", "pnl"].values[0]
)

agent_positions = agent_positions[agent_positions["token_id"] != "WITHDRAWAL_SHARE"].sort_values("pnl", ascending=False)
agent_positions.iloc[:10]


# %%
block_range = None
# block_range = [490, 700]

# %% ###### Plot normalized pnl over time ######
import seaborn as sns

if block_range is not None:
    plot_data = pnl[(pnl["block_number"] >= block_range[0]) & (pnl["block_number"] <= block_range[1])]
else:
    plot_data = pnl
ax = sns.lineplot(data=plot_data, x="block_number", y="normalized_pnl", hue="username")
ax.tick_params(axis="x", rotation=45)

# %% ###### Plot normalized token positions time ######
grouped_positions = (
    positions.groupby(["block_number", "timestamp", "username", "token_type"])
    .agg({"token_balance": "sum"})
    .reset_index()
)

# Normalized positions
grouped_positions["token_max_balance"] = grouped_positions.groupby("token_type")["token_balance"].transform("max")
grouped_positions["normalized_token_balance"] = (
    grouped_positions["token_balance"] / grouped_positions["token_max_balance"]
)

if block_range is not None:
    plot_data = grouped_positions[
        (grouped_positions["block_number"] >= block_range[0]) & (grouped_positions["block_number"] <= block_range[1])
    ]
else:
    plot_data = grouped_positions

ax = sns.lineplot(
    data=plot_data[plot_data["username"] == "agent0"],
    x="block_number",
    y="normalized_token_balance",
    hue="token_type",
)
ax.tick_params(axis="x", rotation=45)

# %% ###### Plot fixed rates ######

if block_range is not None:
    plot_data = pool_info[(pool_info["block_number"] >= block_range[0]) & (pool_info["block_number"] <= block_range[1])]
else:
    plot_data = pool_info

ax = sns.lineplot(
    data=pool_info,
    x="block_number",
    y="fixed_rate",
)
ax.tick_params(axis="x", rotation=45)


# %%
