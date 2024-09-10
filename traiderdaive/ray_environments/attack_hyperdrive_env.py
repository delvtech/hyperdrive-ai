"""A hyperdrive rl gym environment."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Iterable

import numpy as np
from agent0 import LocalChain, LocalHyperdrive, PolicyZoo
from fixedpointmath import FixedPoint
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from scipy.special import expit

from .ray_hyperdrive_env import AGENT_PREFIX, POLICY_PREFIX, RayHyperdriveEnv, TradeTypes

# Global suppression of warnings, TODO fix
warnings.filterwarnings("ignore")


class AttackHyperdriveEnv(RayHyperdriveEnv):
    """A simple hyperdrive environment that allows for 2 positions, long and short."""

    def _apply_action(self, agent_id: str, action: np.ndarray) -> list[bool]:
        """Execute the bot action on-chain.

        Arguments
        ---------
        agent_id: str
            Unique identifying string for the agent.
        action: np.ndarray
            Action activations returned by the policy network.

        Returns
        -------
        bool
            True if the trade was successful, False otherwise.
        """
        # TODO
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-nested-blocks
        # pylint: disable=too-many-statements
        trade_success = [
            True,
        ] * (len(TradeTypes))

        # The actual min txn amount is a function of pool state. Without helper functions, we simply add a safe amount.
        min_tx_amount = self.interactive_hyperdrive.config.minimum_transaction_amount * FixedPoint("2")

        long_short_actions = action[:-4]
        long_short_actions = long_short_actions.reshape((len(TradeTypes), self.env_config.max_positions_per_type + 2))
        close_long_short_actions = long_short_actions[:, :-2]
        open_long_short_actions = long_short_actions[:, -2:]

        # Then we can sort by those and the agent can specify the order of trades.
        # The RL bot handles trades in this order:
        # (1) Close long tokens
        # (2) Close short tokens
        # (2) Open long tokens
        # (4) Open short tokens

        # Get agent positions once so we don't hit the database too many times.
        # The downside of this is we could undershoot the max_positions_per_type
        # since any close actions this round will not be accounted for. This
        # is a fine tradeoff, though, since it's an undershoot and the next time
        # apply_action is called the previous closes will be accounted for.
        agent_positions = self.rl_agents[agent_id].get_positions(coerce_float=False)

        # Closing trades
        close_trade_success = self._apply_close_trades(agent_id, close_long_short_actions, agent_positions)
        trade_success[: len(close_trade_success)] = close_trade_success
        # Open trades
        open_trade_success = self._apply_open_trades(agent_id, min_tx_amount, open_long_short_actions, agent_positions)
        trade_success[len(close_trade_success) : len(open_trade_success)] = open_trade_success

        return trade_success

    def create_action_space(self) -> None:
        """Create the action space object & assign it to self."""
        # (longs, shorts) -> close_order_i(logit), new_order(logit), volume)
        self._action_space_in_preferred_format = True
        self.action_space = spaces.Dict(
            {
                agent_id: spaces.Box(
                    low=-1e2,
                    high=1e2,
                    dtype=np.float64,
                    shape=(len(TradeTypes) * (self.env_config.max_positions_per_type + 2),),
                )
                for agent_id in self.agents
            }
        )

    def step(
        self, action_dict: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]]:
        """Takes a step in the the environment. Modified to showcase attack situation.

        Arguments
        ---------
        action: ActType
            An action provided by the agent to update the environment state

        Returns
        -------
        tuple[np.ndarray, float, bool, bool, dict[str, Any]]
            Contains the following

            observation: ObsType
                An element of the environment's observation_space.
            reward: float
                Reward for taking the action.
            terminated: bool
                Whether the agent reaches the terminal state, which can be positive or negative.
                If true, user needs to call reset
            truncated: bool
                Whether the truncation condition outside the scope of the MDP is satisfied,
                e.g., timelimit, or agent going out of bounds.
                If true, user needs to call reset
            info: dict[str, Any]
                Contains auxiliary diagnostic information for debugging, learning, logging.
        """
        # TODO: Verify that env_config.episode length is working
        # TODO: _apply_action() is per agent_id, but _get_observations() is for all agents. Make this consistent?
        # TODO: Verify that truncated/terminated are being used correctly here. Do we need self.terminateds?
        self.logger.info(f"\nStep {self._step_count} Time: {datetime.now().strftime('%I:%M:%S %p')}")
        # Do actions and get truncated status for agents provided, and set the rest to True
        self.interactive_hyperdrive.sync_database()

        # Run apply action twice
        for agent_id, action in action_dict.items():
            _ = self._apply_action(agent_id, action)

        for agent_id, action in action_dict.items():
            _ = self._apply_action(agent_id, action)

        # Run other bots
        # Suppress logging here
        for random_bot in self.random_bots:
            try:
                random_bot.execute_policy_action()
            except BaseException as err:  # pylint: disable=broad-except
                self.logger.warning(f"Failed to execute random bot: {repr(err)}")
                # We ignore errors in random bots
                continue

        # We minimize time between bot making an action, so we advance time after actions have been made
        # but before the observation
        self.chain.advance_time(self.env_config.step_advance_time, create_checkpoints=True)

        self.interactive_hyperdrive.sync_database()

        observations = self._get_observations(agents=action_dict.keys())
        info = self._get_info(agents=action_dict.keys())
        step_rewards = self._calculate_rewards(agents=action_dict.keys())

        episode_over = self._step_count >= self.env_config.episode_length - 1

        truncateds = {agent_id: episode_over for agent_id in action_dict.keys()}
        terminateds = {agent_id: False for agent_id in action_dict.keys()}

        self.truncateds.update({agent_id for agent_id, truncated in truncateds.items() if truncated})
        self.terminateds.update({agent_id for agent_id, terminated in terminateds.items() if terminated})

        truncateds["__all__"] = len(self.truncateds) == len(self.agents)
        terminateds["__all__"] = len(self.terminateds) == len(self.agents)

        self._step_count += 1
        # TODO when does the episode stop?
        return observations, step_rewards, terminateds, truncateds, info
