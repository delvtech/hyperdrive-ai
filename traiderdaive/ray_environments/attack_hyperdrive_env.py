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
    """A hyperdrive environment catered to discovering the burrito attack vector.

    Basic burrito attack steps per episode:
      1. Manual agent LPs
      2. RL agent opens a max short
      3. RL agent LPs
      4. RL agent opens a max long
      5. Time elapses (x-axis amount in plot)
      6. RL agent closes all positions
      7. Manual agent closes all positions

    An more difficult iterative version also exists, whereby time elapses by an
    amount determined by the RL agent:
      1. Manual agent LPs
      2. for iteration in range(num_iterations):
         OR
         while(manual_agent_lp_balance > 0):
           2.1. RL agent opens a max short
           2.2. RL agent LPs
           2.3. RL agent opens a max long
           2.4. RL agent chooses not to trade for X steps
               - the optimal X is determined by the pool config & agent budgets
           2.5. RL agent closes all positions

      3. Manual agent closes all positions (potentially receiving nothing)
    """

    def __init__(self, env_config) -> None:
        # For this attack we only want one agent and zero random bots
        assert self.env_config.num_random_bots == 0
        assert self.env_config.num_random_hold_bots == 0
        assert self.env_config.num_agents == 1
        # The attack assumes they only hold 1 trade of each type
        # Note that the bot _could_ learn this; we are enforcing it to help out
        assert self.env_config.max_positions_per_type == 1
        super().__init__(env_config)

    def create_action_space(self) -> None:
        """Returns the action space object."""
        # Allowing the agent to propose 3 actions per step instead of 1
        self._action_space_in_preferred_format = True
        self.action_space = spaces.Dict(
            {
                agent_id: spaces.Box(
                    low=-1e2,
                    high=1e2,
                    dtype=np.float64,
                    shape=(3 * len(TradeTypes) * (self.env_config.max_positions_per_type + 2),),
                )
                for agent_id in self.agents
            }
        )
        
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

<<<<<<<<<<<<<<  ✨ Codeium Command 🌟  >>>>>>>>>>>>>>>>
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
<<<<<<<  c40afb40-be44-41b7-ba88-be13d033a012  >>>>>>>
