"""A hyperdrive rl gym environment."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from fixedpointmath import FixedPoint
from gymnasium import spaces

from .ray_hyperdrive_env import RayHyperdriveEnv, TradeTypes

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

    @dataclass(kw_only=True)
    class Config(RayHyperdriveEnv.Config):
        """The configuration for AttackHyperdriveEnv."""

        num_trade_sets_per_step: int = 3

    def __init__(self, env_config) -> None:
        super().__init__(env_config)
        # For this attack we only want one agent and zero random bots
        assert self.env_config.num_random_bots == 0
        assert self.env_config.num_random_hold_bots == 0
        assert self.env_config.num_agents == 1
        # The attack assumes they only hold 1 trade of each type
        # Note that the bot _could_ learn this; we are enforcing it to help out
        assert self.env_config.max_positions_per_type == 1

    def create_action_space(self) -> None:
        """Returns the action space object."""
        # Allowing the agent to propose 3 actions per step instead of 1
        assert isinstance(self.env_config, self.Config)  # narrow type
        self._action_space_in_preferred_format = True
        self.action_space = spaces.Dict(
            {
                agent_id: spaces.Box(
                    low=-1e2,
                    high=1e2,
                    dtype=np.float64,
                    shape=(self.env_config.num_trade_sets_per_step * self.action_length_per_trade_set,),
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

        assert isinstance(self.env_config, self.Config)  # narrow type

        # The agent can now specify 3 trade sets.
        # For each set they can execute 3 trades, but we expect them to learn
        # to pick one tarde per set.

        trade_success = [
            True,
        ] * (self.env_config.num_trade_sets_per_step * len(TradeTypes))

        agent_positions = self.rl_agents[agent_id].get_positions(coerce_float=False)
        # The actual min txn amount is a function of pool state. Without helper functions, we simply add a safe amount.
        min_tx_amount = self.interactive_hyperdrive.config.minimum_transaction_amount * FixedPoint("2")

        for trade_idx in range(self.env_config.num_trade_sets_per_step):
            sub_trade_success = [
                True,
            ] * len(TradeTypes)
            start_idx = trade_idx * self.action_length_per_trade_set
            end_idx = start_idx + self.action_length_per_trade_set
            assert end_idx <= len(action), "Indexing out of bounds."

            trade_action = action[start_idx:end_idx]

            long_short_actions = trade_action[:-4]
            long_short_actions = long_short_actions.reshape(
                (len(TradeTypes), self.env_config.max_positions_per_type + 2)
            )
            close_long_short_actions = long_short_actions[:, :-2]
            open_long_short_actions = long_short_actions[:, -2:]
            lp_actions = trade_action[-4:]

            # Closing trades
            close_trade_success = self._apply_close_trades(agent_id, close_long_short_actions, agent_positions)
            sub_trade_success[: len(close_trade_success)] = close_trade_success

            # Open trades
            open_trade_success = self._apply_open_trades(
                agent_id, min_tx_amount, open_long_short_actions, agent_positions
            )
            sub_trade_success[len(close_trade_success) : len(open_trade_success)] = open_trade_success

            # LP trade
            sub_trade_success[-1] = self._apply_lp_trades(agent_id, min_tx_amount, lp_actions)

            # Record success list
            trade_success.extend(sub_trade_success)

        return trade_success
