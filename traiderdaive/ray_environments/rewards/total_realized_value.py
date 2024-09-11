"""Reward that uses the realized value plus eth balance per episode."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from agent0.ethpy.base import get_account_balance

from .base_reward import BaseReward


class TotalRealizedValue(BaseReward):

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
        agent_positions = current_positions[current_positions["wallet_address"] == self.env.rl_agents[agent_id].address]
        # We use the absolute realized value and the eth balance as the reward
        total_realized_value = float(agent_positions["realized_value"].sum())
        agent_eth_balance = get_account_balance(self.env.chain._web3, self.env.rl_agents[agent_id].address)
        assert agent_eth_balance is not None  # type narrowing
        reward = total_realized_value + agent_eth_balance
        return reward
