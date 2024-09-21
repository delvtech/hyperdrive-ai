"""Reward that uses the realized value plus eth balance per episode."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from agent0.ethpy.base import get_account_balance

from .base_reward import BaseReward


class ValueAboveInitialLp(BaseReward):

    def calculate_rewards(self, agents: Iterable[str] | None = None) -> dict[str, float]:
        """Computes the rewards for all agents.

        Arguments
        ---------
        agents: Iterable[str] | None
            List of agent IDs. If None, calculate for all agents.

        Returns
        -------
        float
            The computed reward.
        """
        return self._calculate_rewards_per_episode(agents)

    def calculate_agent_reward(self, agent_id: str, current_positions: pd.DataFrame) -> float:
        """Computes the Realized Value reward for the given agent.

        Arguments
        ---------
        agent_id: str
            The ID of the agent for which to compute the reward.
        current_positions: pd.DataFrame
            The current positions of the agents as returned by agent0.

        Returns
        -------
        float
            The computed reward.
        """
        # The "total value" is the absolute realized value plus the eth balance
        # Get total value of initial LP address
        initial_lp_address = self.env.chain.get_deployer_address()
        lp_positions = current_positions[current_positions["wallet_address"] == initial_lp_address]
        lp_realized_value = float(lp_positions["realized_value"].sum())
        lp_eth_balance = get_account_balance(self.env.chain._web3, initial_lp_address)
        assert lp_eth_balance is not None  # type narrowing
        lp_total_value = lp_realized_value + lp_eth_balance
        # Get total value of agent address
        agent_positions = current_positions[current_positions["wallet_address"] == self.env.rl_agents[agent_id].address]
        total_realized_value = float(agent_positions["realized_value"].sum())
        agent_eth_balance = get_account_balance(self.env.chain._web3, self.env.rl_agents[agent_id].address)
        assert agent_eth_balance is not None  # type narrowing
        agent_total_value = total_realized_value + agent_eth_balance
        # Reward is the difference between these two total values
        return agent_total_value - lp_total_value
