"""The gym environments for hyperdrive. Also registers the environment to gym."""

from gymnasium.envs.registration import register

from .gym_environments.gym_async_full_hyperdrive_env import AsyncFullHyperdriveEnv
from .gym_environments.gym_full_hyperdrive_env import FullHyperdriveEnv
from .gym_environments.gym_simple_hyperdrive_env import SimpleHyperdriveEnv

# TODO expose ray environments when ready

# Register hyperdrive envs to gym
register(
    id="traiderdaive/gym_simple_hyperdrive_env",
    entry_point="traiderdaive:SimpleHyperdriveEnv",
    max_episode_steps=1000,
)

register(
    id="traiderdaive/gym_full_hyperdrive_env",
    entry_point="traiderdaive:FullHyperdriveEnv",
    max_episode_steps=1000,
)

register(
    id="traiderdaive/gym_async_full_hyperdrive_env",
    entry_point="traiderdaive:AsyncFullHyperdriveEnv",
    max_episode_steps=1000,
)
