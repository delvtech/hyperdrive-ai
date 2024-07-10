"""The gym environments for hyperdrive. Also registers the environment to gym."""

from gymnasium.envs.registration import register

from .gym_environments.gym_full_hypedrive_env import FullHyperdriveEnv
from .gym_environments.gym_simple_hyperdrive_env import SimpleHyperdriveEnv

# TODO expose ray environments when ready

# Register hyperdrive envs to gym
register(
    id="traiderdaive/simple_hyperdrive_env",
    entry_point="traiderdaive.gym_environments.simple_hyperdrive_env:SimpleHyperdriveEnv",
    max_episode_steps=1000,
)

register(
    id="traiderdaive/full_hyperdrive_env",
    entry_point="traiderdaive.gym_environments.full_hyperdrive_env:FullHyperdriveEnv",
    max_episode_steps=1000,
)
