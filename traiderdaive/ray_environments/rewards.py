from typing import Iterable

from agent0.ethpy.base import get_account_balance


def calculate_delta_pnl(thing, agents: Iterable[str] | None = None) -> dict[str, float]:
    agents = agents or thing.agents
    # The total delta for this episode

    current_positions = thing.interactive_hyperdrive.get_positions(
        show_closed_positions=True, calc_pnl=True, coerce_float=True
    )
    reward = {}
    for agent_id in agents:
        # Filter by agent ID
        agent_positions = current_positions[current_positions["wallet_address"] == thing.rl_agents[agent_id].address]
        # Reward is in units of base
        # We use the change in pnl as the reward
        # The agent_positions shows the pnl of all positions
        # Sum across all positions
        # TODO one option here is to only look at base positions instead of sum across all positions.
        # TODO handle the case where pnl calculation doesn't return a number
        # when you can't close the position
        new_pnl = float(agent_positions["pnl"].sum())
        reward[agent_id] = new_pnl - thing._prev_pnls[agent_id]
        thing._prev_pnls[agent_id] = new_pnl
    return reward


def calculate_realized_value(thing, agents: Iterable[str] | None = None) -> dict[str, float]:
    agents = agents or thing.agents
    # The total delta for this episode
    reward = {}
    if thing._step_count == thing.env_config.episode_length - 1:
        current_positions = thing.interactive_hyperdrive.get_positions(
            show_closed_positions=True, calc_pnl=True, coerce_float=True
        )
        for agent_id in agents:
            # Filter by agent ID
            agent_positions = current_positions[
                current_positions["wallet_address"] == thing.rl_agents[agent_id].address
            ]
            total_value = float(agent_positions["realized_value"].sum())
            # reward is in units of base
            reward[agent_id] = total_value
    else:
        reward = {agent_id: 0.0 for agent_id in agents}
    return reward


def calculate_total_pnl(thing, agents: Iterable[str] | None = None) -> dict[str, float]:
    agents = agents or thing.agents
    # The total delta for this episode
    reward = {}
    if thing._step_count == thing.env_config.episode_length - 1:
        current_positions = thing.interactive_hyperdrive.get_positions(
            show_closed_positions=True, calc_pnl=True, coerce_float=True
        )
        for agent_id in agents:
            # Filter by agent ID
            agent_positions = current_positions[
                current_positions["wallet_address"] == thing.rl_agents[agent_id].address
            ]
            total_pnl = float(agent_positions["pnl"].sum())
            # reward is in units of base
            reward[agent_id] = total_pnl
    else:
        reward = {agent_id: 0.0 for agent_id in agents}
    return reward


def calculate_total_value(thing, agents: Iterable[str] | None = None) -> dict[str, float]:
    agents = agents or thing.agents
    # The total delta for this episode
    reward = {}
    if thing._step_count == thing.env_config.episode_length - 1:
        current_positions = thing.interactive_hyperdrive.get_positions(
            show_closed_positions=True, calc_pnl=True, coerce_float=True
        )
        for agent_id in agents:
            # Filter by agent ID
            agent_positions = current_positions[
                current_positions["wallet_address"] == thing.rl_agents[agent_id].address
            ]
            # We use the absolute realized value and the eth balance as the reward
            total_realized_value = float(agent_positions["realized_value"].sum())
            agent_eth_balance = get_account_balance(thing.chain._web3, thing.rl_agents[agent_id].address)
            assert agent_eth_balance is not None  # type narrowing
            reward[agent_id] = total_realized_value + agent_eth_balance
    else:
        reward = {agent_id: 0.0 for agent_id in agents}
    return reward
