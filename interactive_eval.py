import logging

import argparse
import sys
from functools import partial
from IPython import embed
from typing import Sequence

import numpy as np
import ray
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig

from agent0 import LocalChain, LocalHyperdrive
from agent0.core.hyperdrive.interactive.local_hyperdrive_agent import LocalHyperdriveAgent
from fixedpointmath import FixedPoint
from traiderdaive.ray_environments.attack_hyperdrive_env import AttackHyperdriveEnv
from traiderdaive.ray_environments.ray_hyperdrive_env import AGENT_PREFIX, POLICY_PREFIX
from traiderdaive.ray_environments.rewards import TotalRealizedValue
from traiderdaive.ray_environments.variable_rate_policy import ConstantVariableRate


def main():
    """Burrito evaluation main entry point."""
    parsed_args = parse_arguments(sys.argv)

    reward = TotalRealizedValue
    rate_policy = ConstantVariableRate()

    env_config = AttackHyperdriveEnv.Config(
        variable_rate_policy=rate_policy,
        reward_policy=reward,
        num_random_bots=0,
        num_random_hold_bots=0,
        num_agents=1,
        episode_length=50 + 1,  # +1 Because the final sample() call calls reset()
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
        .evaluation(
            evaluation_duration=1,
            evaluation_duration_unit="episodes",
            # evaluation_config=AlgorithmConfig.overrides(explore=False),
        )
    )
    algo = config.build()
    algo.restore(parsed_args.checkpoint_dir)
    # TODO: sanitize checkpoint path

    def step(num_steps=1, explore=False, return_samples=False):
        """Runs a single step in the environment."""
        samples = algo.env_runner.sample(num_timesteps=num_steps, explore=explore)
        if return_samples:
            return samples

    num_steps = (
        parsed_args.breakpoint_step if parsed_args.breakpoint_step is not None else env_config.episode_length - 1
    )
    # Run custom number of steps:
    step(num_steps=num_steps, explore=False)

    # Get underlying agent0 objects
    chain: LocalChain = algo.env_runner.env.env.chain
    pool: LocalHyperdrive = chain._deployed_hyperdrive_pools[0]
    agent: LocalHyperdriveAgent = chain._chain_agents[0]

    get_pool_info = partial(pool.get_pool_info, coerce_float=True)  # lol

    def _get_timestamp_lookup():
        pool_info = get_pool_info()
        # Get mapping from block to timestamp
        timestamp_lookup = pool_info[
            [
                "block_number",
                "timestamp",
                "epoch_timestamp",
            ]
        ]
        return timestamp_lookup

    def get_trade_events():
        # Modify and get relevant columns of dataframes
        trade_events = pool.get_trade_events(coerce_float=True).merge(right=_get_timestamp_lookup(), on="block_number")[
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
        return trade_events

    def get_positions():
        positions = pool.get_historical_positions(coerce_float=True).merge(
            right=_get_timestamp_lookup(), on="block_number"
        )[
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
        return positions

    def get_pnl():
        pnl = pool.get_historical_pnl(coerce_float=True).merge(right=_get_timestamp_lookup(), on="block_number")[
            ["block_number", "timestamp", "username", "pnl", "epoch_timestamp"]
        ]
        # Add normalized pnl
        pnl["budget"] = np.nan
        pnl.loc[pnl["username"] != "agent0", "budget"] = float(FixedPoint(100_000))
        pnl.loc[pnl["username"] == "agent0", "budget"] = float(env_config.rl_agent_budget)

        pnl["normalized_pnl"] = pnl["pnl"] / pnl["budget"]

        return pnl

    def get_agent_positions():
        ###### Get and prepare final positions of bad agent ######
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

        agent_positions = agent_positions[agent_positions["token_id"] != "WITHDRAWAL_SHARE"].sort_values(
            "pnl", ascending=False
        )
        return agent_positions.iloc[:10]

    def help():
        line = "\n" + "-" * 20 + "\n"
        nl = "\n"
        tab = " " * 4
        help = (
            f"{line}Agent0 Objects:{line}"
            f"chain: LocalChain{nl}"
            f"pool: LocalHyperdrive{nl}"
            f"agent: LocalHyperdriveAgent{nl}"
            f"{line}Eval Functions:{line}"
            f"step(num_steps=1):{nl}{tab}Take N steps in the environment{nl}"
            f"get_pool_info():{nl}{tab}Returns DataFrame with pool info.{nl}"
            f"get_trade_events():{nl}{tab}Returns DataFrame with all trade events.{nl}"
            f"get_positions():{nl}{tab}Returns DataFrame with all positions.{nl}"
            f"get_pnl():{nl}{tab}Returns agent's pnl.{nl}"
            f"get_agent_positions():{nl}{tab}Returns agent positions (for plotting).{nl}"
            f"help():{nl}{tab}Print this help block.{nl}"
            f"{line}Usage:{line}"
            f"Run one step:{nl}{tab}In [1]: step(){nl}"
            f"Run 50 steps:{nl}{tab}In [1]: step(50){nl}"
            f"View pool info:{nl}{tab}In [1]: get_pool_info(){nl}"
            f"View trade events:{nl}{tab}In [1]: get_trade_events(){nl}"
            f"Save trade events as csv:{nl}{tab}In [1]: get_trade_events().to_csv('<path>/<to>/trade_events.csv'){nl}"
        )
        print(help)
    help()

    embed(using=False)
    raise SystemExit


def parse_arguments(argv: Sequence[str] | None = None):
    """Parses input arguments."""
    parser = argparse.ArgumentParser(description="Interactive Burrito Evaluation")
    parser.add_argument(
        "--checkpoint-dir",
        "-c",
        type=str,
        default=".",
        help="Path to where checkpoint",
    )
    parser.add_argument(
        "--breakpoint-step",
        "-b",
        type=int,
        default=None,
        help="Episode steps to run before halting.",
    )
    # Use system arguments if none were passed
    if argv is None:
        argv = sys.argv
    return parser.parse_args()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARNING)
    main()
