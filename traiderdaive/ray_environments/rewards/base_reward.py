"""Base class for specifying hyperdrive trading agent rewards."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from ..ray_hyperdrive_env import RayHyperdriveEnv


class BaseReward:
    """Base reward class for ray hyperdrive environments.

    This class can be subclassed to implement custom reward functions.
    All subclasses should implement the `calculate_agent_reward()`
    and the calculate_rewards() methods.
    """

    def __init__(self, env: RayHyperdriveEnv):
        """Initializes the Reward class with the environment.

        Arguments
        ---------
        env: Any
            The environment instance.
        """
        self.env = env

    def calculate_rewards(self, agents: Iterable[str] | None = None) -> dict[str, float]:
        """Computes the rewards for all agents.

        This is the primary method that is called by the environment and should
        be overridden by subclasses.

        Arguments
        ---------
        agents: Iterable[str] | None
            List of agent IDs. If None, calculate for all agents.

        Returns
        -------
        float
            The computed reward.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def calculate_agent_reward(self, agent_id: str, current_positions: pd.DataFrame) -> float:
        """Computes the reward for the given agent.
        This method should be overridden by subclasses.

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
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _calculate_rewards_per_step(self, agents: Iterable[str] | None = None) -> dict[str, float]:
        """Calculates rewards each step for all agents.

        Arguments
        ---------
        agents: Iterable[str] | None
            List of agent IDs. If None, calculate for all agents.

        Returns
        -------
        dict[str, float]
            A dictionary mapping agent IDs to their corresponding rewards.
        """
        if agents is None:
            agents = self.env.agents

        current_positions = self.env.interactive_hyperdrive.get_positions(
            show_closed_positions=True, calc_pnl=True, coerce_float=True
        )

        rewards = {}
        for agent_id in agents:
            rewards[agent_id] = self.calculate_agent_reward(agent_id, current_positions)

        return rewards

    def _calculate_rewards_per_episode(self, agents: Iterable[str] | None = None) -> dict[str, float]:
        """Calculates rewards at the end of an episode for all agents.

        Arguments
        ---------
        agents: Iterable[str] | None
            List of agent IDs. If None, calculate for all agents.

        Returns
        -------
        dict[str, float]
            A dictionary mapping agent IDs to their corresponding rewards.
        """
        if agents is None:
            agents = self.env.agents

        rewards = {}
        if self.env._step_count == self.env.env_config.episode_length - 1:
            current_positions = self.env.interactive_hyperdrive.get_positions(
                show_closed_positions=True, calc_pnl=True, coerce_float=True
            )
            for agent_id in agents:
                rewards[agent_id] = self.calculate_agent_reward(agent_id, current_positions)
        else:
            rewards = {agent_id: 0.0 for agent_id in agents}

        return rewards
