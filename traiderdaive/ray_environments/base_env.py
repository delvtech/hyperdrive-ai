"""A hyperdrive rl gym environment."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Type

import numpy as np
from agent0 import LocalChain, LocalHyperdrive, PolicyZoo
from fixedpointmath import FixedPoint
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from scipy.special import expit

from .rewards import DeltaPnl
from .variable_rate_policy import RandomNormalVariableRate, VariableRatePolicy

if TYPE_CHECKING:
    from .rewards.base_reward import BaseReward

AGENT_PREFIX = "agent"
POLICY_PREFIX = "policy"

# Global suppression of warnings, TODO fix
warnings.filterwarnings("ignore")


class BaseEnv(MultiAgentEnv):
    """A simple hyperdrive environment that allows for 2 positions, long and short."""

    # pylint: disable=too-many-instance-attributes

    @dataclass(kw_only=True)
    class Config:
        """The configuration for base env."""

        # How to render the environment
        # TODO figure out what this does
        render_mode: str | None = None

        # Experiment Config
        position_reward_scale: float = 1
        # Number of RayHyperdriveEnv steps per episode
        episode_length: int = 50
        # Number of episodes for each training step
        num_episodes_per_update: int = 5
        # Number of training iterations (after rollouts have been collected)
        num_epochs_sgd: int = 10
        # Number of interations of the full train loop (collecting rollouts & training model)
        # One set of rollouts is {num_episodes_per_update * episode_length} env steps
        # One training loop is {num_epochs_sgd} iterations training on rollouts
        num_training_loops: int = 100000

        # How much to advance time per step
        step_advance_time: int = 8 * 3600  # 8 hours

        # RL Agents Config
        num_agents: int = 4
        # The constant trade amounts for longs and shorts
        rl_agent_budget: FixedPoint = FixedPoint(1_000_000)
        max_positions_per_type: int = 10
        base_reward_scale: float = 0.0
        # The threshold for the probability of opening and closing orders
        open_threshold: float = 0.5
        close_threshold: float = 0.5

        # TODO: Check if PPO is already sampling actions!
        sample_actions: bool = False
        # Sets alternate ports for eval to avoid connecting to a training chain
        eval_mode: bool = False

    # Defines allowed render modes and fps
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def init_config(self, env_config):
        if env_config.get("env_config") is None:
            self.env_config = self.Config()
        else:
            self.env_config = env_config["env_config"]
            assert isinstance(self.env_config, BaseEnv.Config)

    # FIXME give type to env_config
    def __init__(
        self,
        env_config,
    ):
        """Initializes the environment"""
        self.worker_index = env_config.worker_index

        self.init_config(env_config)

        # Multiagent setup
        self._terminateds = set()
        self._truncateds = set()

        self.eval_mode = self.env_config.eval_mode
        self.sample_actions = self.env_config.sample_actions
        if self.eval_mode:
            db_port = 5434
            chain_port = 10001
        else:
            db_port = 5435 + self.worker_index
            chain_port = 10002 + self.worker_index

        local_chain_config = LocalChain.Config(
            block_timestamp_interval=12,
            db_port=db_port,
            chain_port=chain_port,
            calc_pnl=self.eval_mode,
            manual_database_sync=(not self.eval_mode),
            backfill_pool_info=False,
        )

        self.chain = LocalChain(local_chain_config)

        # Instantiate the random number generator
        random_seed = np.random.randint(1, 99999) + self.worker_index
        self.rng = np.random.default_rng(random_seed)

        # Define the rl agents
        self._agent_ids = {f"{AGENT_PREFIX}{i}" for i in range(self.env_config.num_agents)}
        self.agents = {
            agent_id: self.chain.init_agent(
                base=self.env_config.rl_agent_budget,
                eth=FixedPoint(100),
                name=agent_id,
            )
            for agent_id in self._agent_ids
        }

        # Call any initialization setup before we snapshot
        self.setup_environment()

        # Save a snapshot of initial conditions for resets
        self.chain.save_snapshot()

        assert self.env_config.render_mode is None or self.env_config.render_mode in self.metadata["render_modes"]
        self.render_mode = self.env_config.render_mode

        self.action_space = spaces.Dict({agent_id: self.create_action_space() for agent_id in self._agent_ids})
        self.observation_space = self.create_observation_space()

        # These variables are needed by the base class
        self._action_space_in_preferred_format = True
        self._obs_space_in_preferred_format = True

        # episode variables
        self._prev_pnls: dict[str, float] = {agent_id: 0.0 for agent_id in self._agent_ids}
        self._step_count = 0

        # setup logger
        self.logger = logging.getLogger()
        self.logger.info("rng seed: " + str(random_seed))
        super().__init__()

    def setup_environment(self):
        """Function to run initial setup of the RL environment. For example,
        this is where we can fund agents.
        """
        raise NotImplementedError

    def create_action_space(self) -> spaces.Box:
        """Function to create the action space for the environment."""
        raise NotImplementedError

    def create_observation_space(self) -> spaces.Dict:
        """Function to create the observation space for the environment."""
        raise NotImplementedError

    def __del__(self) -> None:
        self.chain.cleanup()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Resets the environment to an initial internal state.

        Arguments
        ---------
        seed: int | None
            The seed to initialize the random generator to pass for each bot
        options: dict[str, Any] | None
            Additional information to specify how the environment is reset (optional,
            depending on the specific environment)

        Returns
        -------
        tuple[np.ndarray, dict[str, Any]]
            The observation and info from the environment
        """

        # TODO do random seeds properly
        super().reset(seed=seed)

        # TODO randomize pool parameters here
        # We can do this by deploying a new pool
        # For now, we use a single pool with default parameters
        # and use snapshotting to reset

        # Load the snapshot for initial conditions
        self.chain.load_snapshot()

        self.reset_env()

        # Reset internal member variables
        self._prev_pnls: dict[str, float] = {agent_id: 0.0 for agent_id in self._agent_ids}
        self._step_count = 0
        self._terminateds = set()
        self._truncateds = set()

        # Get first observation and info
        observations = self._get_observations()
        info = self._get_info()

        return observations, info

    def reset_env(self):
        """Resets any custom state environment."""
        raise NotImplementedError

    def step(
        self, action_dict: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]]:
        """Takes a step in the the environment.

        .. note::
        Truncated & terminated result in different loss updates for the reward
        estimator network. In our case, the environment represents an
        infinite-horizon (continuing) task. The goal for our agent is to
        maximize the cumulative reward over an infinite or indefinite time
        horizon.

        This means we need to include a discount factor to ensure convergence.
        As such, we _always_ want `terminated == False` -- aka the game never
        ends ("terminates"). We do need to accumulate gradients and do model
        updates, however, so we must discretize the environment and impose some
        sort of stopping criteria. This can be achieved by truncating, where we
        stop the game at some time (can be arbitrary) that is not known by the
        agent (i.e. not in the observation space).

        Arguments
        ---------
        action: ActType
            An action provided by the agent to update the environment state

        Returns
        -------
        tuple[np.ndarray, float, bool, bool, dict[str, Any]]
            Contains the following

            observation: ObsType
                An element of the environment's observation_space.
            reward: float
                Reward for taking the action.
            terminated: bool
                Whether the agent reaches the terminal state, which can be positive or negative.
                If true, user needs to call reset
            truncated: bool
                Whether the truncation condition outside the scope of the MDP is satisfied,
                e.g., timelimit, or agent going out of bounds.
                If true, user needs to call reset
            info: dict[str, Any]
                Contains auxiliary diagnostic information for debugging, learning, logging.
        """
        # TODO: Verify that env_config.episode length is working
        # TODO: _apply_action() is per agent_id, but _get_observations() is for all agents. Make this consistent?
        # TODO: Verify that truncated/terminated are being used correctly here. Do we need self.terminateds?
        self.logger.info(f"\nStep {self._step_count} Time: {datetime.now().strftime('%I:%M:%S %p')}")
        # Do actions and get truncated status for agents provided, and set the rest to True
        self.interactive_hyperdrive.sync_database()

        for agent_id, action in action_dict.items():
            _ = self._apply_action(agent_id, action)

        # Run other bots
        # Suppress logging here
        for random_bot in self.random_bots:
            try:
                random_bot.execute_policy_action()
            except BaseException as err:  # pylint: disable=broad-except
                self.logger.warning(f"Failed to execute random bot: {repr(err)}")
                # We ignore errors in random bots
                continue

        # We minimize time between bot making an action, so we advance time after actions have been made
        # but before the observation
        self.chain.advance_time(self.env_config.step_advance_time, create_checkpoints=True)

        # Update variable rate with probability Config.rate_change_probability
        # TODO: Parameterize distribution and clip
        if self.env_config.variable_rate_policy.do_change_rate(self.rng):
            new_rate = self.env_config.variable_rate_policy.get_new_rate(
                self.interactive_hyperdrive.interface, self.rng
            )
            self.interactive_hyperdrive.set_variable_rate(new_rate)

        self.interactive_hyperdrive.sync_database()

        observations = self._get_observations(agents=action_dict.keys())
        info = self._get_info(agents=action_dict.keys())
        step_rewards = self._calculate_rewards(agents=action_dict.keys())

        episode_over = self._step_count >= self.env_config.episode_length - 1

        truncateds = {agent_id: episode_over for agent_id in action_dict.keys()}
        terminateds = {agent_id: False for agent_id in action_dict.keys()}

        self._truncateds.update({agent_id for agent_id, truncated in truncateds.items() if truncated})
        self._terminateds.update({agent_id for agent_id, terminated in terminateds.items() if terminated})

        truncateds["__all__"] = len(self._truncateds) == len(self._agent_ids)
        terminateds["__all__"] = len(self._terminateds) == len(self._agent_ids)

        self._step_count += 1
        # TODO when does the episode stop?
        return observations, step_rewards, terminateds, truncateds, info

    def _get_info(self, agents: Iterable[str] | None = None) -> dict:
        agents = agents or self._agent_ids
        info_dict = {agent_id: {} for agent_id in agents}
        return info_dict

    def _get_observations(self, agents: Iterable[str] | None = None) -> dict[str, np.ndarray]:
        agents = agents or self._agent_ids
        # Get the pool config
        pool_config_df = self.interactive_hyperdrive.get_pool_config(coerce_float=True)
        pool_config_df = pool_config_df[self.env_config.pool_config_columns].astype(float)
        pool_config = pool_config_df.values.astype(np.float64)
        # Get the latest pool state feature from the db
        pool_info_df = self.interactive_hyperdrive.get_pool_info(coerce_float=True)
        pool_info_df = pool_info_df[self.env_config.pool_info_columns].iloc[-1].astype(float)
        pool_info = pool_info_df.values.astype(np.float64)
        # Get block timestamp and steps remaining. We convert timestamp to epoch time here
        current_block = self.interactive_hyperdrive.interface.get_current_block()
        timestamp = self.interactive_hyperdrive.interface.get_block_timestamp(current_block)
        block_timestamp = np.array([timestamp], dtype=np.float64)
        steps_remaining = np.array([self.env_config.episode_length - self._step_count], dtype=np.float64)
        # TODO can also add other features, e.g., opening spot price

        out_obs = {}
        for agent_id in agents:
            # Long Features: trade type, order_i -> [volume, value, normalized_time_remaining]
            long_features = np.zeros(self.num_long_features, dtype=np.float64)
            # Short Features: trade type, order_i -> [volume, value, normalized_time_remaining]
            short_features = np.zeros(self.num_short_features, dtype=np.float64)
            # LP: -> [volume, value]
            lp_features = np.zeros(self.num_lp_features, dtype=np.float64)
            # Additional features: pnl
            wallet_features = np.zeros(self.num_wallet_features, dtype=np.float64)

            # Observation data uses floats
            open_agent_positions = self.agents[agent_id].get_positions(coerce_float=True, calc_pnl=True)
            all_agent_positions = self.agents[agent_id].get_positions(
                coerce_float=True, calc_pnl=True, show_closed_positions=True
            )

            if not open_agent_positions.empty:
                position_duration = self.interactive_hyperdrive.config.position_duration
                # We keep negative values for time past maturity
                open_agent_positions["normalized_time_remaining"] = (
                    open_agent_positions["maturity_time"] - timestamp
                ) / position_duration

                long_orders = open_agent_positions[open_agent_positions["token_type"] == "LONG"]
                # Ensure data is the same as the action space
                long_orders = long_orders.sort_values("maturity_time")
                long_orders = long_orders[["token_balance", "pnl", "normalized_time_remaining"]].values.flatten()

                short_orders = open_agent_positions[open_agent_positions["token_type"] == "SHORT"]
                # Ensure data is the same as the action space
                short_orders = short_orders.sort_values("maturity_time")
                short_orders = short_orders[["token_balance", "pnl", "normalized_time_remaining"]].values.flatten()

                lp_orders = open_agent_positions[open_agent_positions["token_type"] == "LP"]
                lp_orders = lp_orders[["token_balance", "pnl"]].values.flatten()

                # Add data to static size arrays
                long_features[: len(long_orders)] = long_orders
                short_features[: len(short_orders)] = short_orders
                lp_features[: len(lp_orders)] = lp_orders
                # Get PNL
                total_pnl = np.array(all_agent_positions["pnl"].sum(), dtype=np.float64)
                wallet_features[0] = total_pnl

            out_obs[agent_id] = np.concatenate(
                [
                    pool_config,
                    pool_info,
                    block_timestamp,
                    steps_remaining,
                    long_features,
                    short_features,
                    lp_features,
                    wallet_features,
                ]
            )

        return out_obs

    def _calculate_rewards(self, agents: Iterable[str] | None = None) -> dict[str, float]:
        return self.reward.calculate_rewards(agents)

    def render(self) -> None:
        """Renders the environment. No rendering available for hyperdrive env."""
        return None
