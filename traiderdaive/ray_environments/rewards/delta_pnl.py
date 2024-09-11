"""Reward that uses the delta PNL per step."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .base_reward import BaseReward


class DeltaPnl(BaseReward):
    def __init__(self, env):
        super().__init__(env)
        self.prev_pnls: dict[str, float] = {agent_id: 0.0 for agent_id in self.env.agents}

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
        return self._calculate_rewards_per_step(agents)

    def calculate_agent_reward(self, agent_id: str, current_positions: pd.DataFrame) -> float:
        """Computes the Delta PnL reward for the given agent.

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
        agent_positions = current_positions[current_positions["wallet_address"] == self.env.rl_agents[agent_id].address]
        # Get new PnL
        new_pnl = float(agent_positions["pnl"].sum())
        # Reward is in units of base
        reward = new_pnl - self.prev_pnls[agent_id]
        # Update prev PnL
        self.prev_pnls[agent_id] = new_pnl
        return reward
