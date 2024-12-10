"""Reward that uses the total PNL per episode."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .base_reward import BaseReward


class TotalPnl(BaseReward):

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
        # Filter by agent ID
        agent_positions = current_positions[current_positions["wallet_address"] == self.env.agents[agent_id].address]
        total_pnl = float(agent_positions["pnl"].sum())
        # reward is in units of base
        return total_pnl
