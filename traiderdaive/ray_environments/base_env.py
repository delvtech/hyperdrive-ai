"""A hyperdrive rl gym environment."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from logging import Logger
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
from agent0 import LocalChain
from fixedpointmath import FixedPoint
from gymnasium import spaces
from numpy.random import Generator
from ray.rllib.env.multi_agent_env import MultiAgentEnv

if TYPE_CHECKING:
    from agent0.core.hyperdrive.interactive.local_hyperdrive_agent import LocalHyperdriveAgent

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

    ######### Public member variables to be used by subclasses ########
    worker_index: int
    """The ray worker index for this instantiation."""
    env_config: Config
    """The environment configuration."""
    rng: Generator
    """The random number generator object."""
    chain: LocalChain
    """The local chain object."""
    agents: dict[str, LocalHyperdriveAgent]
    """The agents in the multiagent environment."""
    step_count: int
    """The current step count in the environment."""
    logger: Logger
    """The logger for the environment."""

    ######### Subclass functions ########
    def init_config(self, env_config) -> None:
        """Function to (1) ensure the env_config is set and typed correctly, and (2)
        set any variable shortcuts needed from the env config.

        Arguments
        ---------
        env_config: Any
            The environment configuration passed in from the runner.
        """
        # TODO there may be a way to implement this in the base class such that
        # self.env_config gets set to the proper config type
        if env_config.get("env_config") is None:
            self.env_config = self.Config()
        else:
            self.env_config = env_config["env_config"]
            assert isinstance(self.env_config, BaseEnv.Config)

    def setup_environment(self) -> None:
        """Function to run initial setup of the RL environment. For example,
        this is where we can fund agents.
        """
        raise NotImplementedError

    def reset_env(self) -> None:
        """Function to run on `reset` that resets any custom state environment.
        Note `load_snapshot` gets called for anything on the chain,
        so this function only needs to reset any custom state managed in the subclass environment.
        E.g., resetting the variable interest policy.
        """
        raise NotImplementedError

    def create_action_space(self) -> spaces.Box:
        """Function to create the action space for a single agent in the environment.

        Returns
        ---------
        spaces.Box
            The action space for a single agent in the environment.
        """
        raise NotImplementedError

    def get_shared_observation(self) -> np.ndarray:
        """Function to gather observations that are shared across all RL agents.

        Returns
        ---------
        np.ndarray
            The 1D shared observations.
        """
        # Defaults to returning empty array
        return np.zeros(shape=(0,))

    def get_agent_observation(self, agent: LocalHyperdriveAgent) -> np.ndarray:
        """Function to gather observations from the environment for the RL agent.

        TODO the agents here are hyperdrive agents, but ideally they would be generic agents
        for all types of environments.

        Arguments
        ---------
        agent: LocalHyperdriveAgent
            The agent to get observations for.

        Returns
        ---------
        np.ndarray
            The 1D observations for the agent.
        """
        # Defaults to returning empty array
        return np.zeros(shape=(0,))

    def apply_action(self, agent: LocalHyperdriveAgent, action: np.ndarray) -> bool:
        """Function to apply an action to the environment.

        Arguments
        ---------
        agent: LocalHyperdriveAgent
            The agent to apply the action to.
        action: np.ndarray
            The action to apply to the agent, encoded as an ndarray, as specified in `create_action_space`.

        Returns
        -------
        bool
            True if the trade was successful, False otherwise.
        """
        raise NotImplementedError

    def step_environment(self):
        """Function to do any updates to the environment after applying actions and advancing time."""
        # Default behavior is no op
        pass

    def calculate_agent_reward(self, agent: LocalHyperdriveAgent) -> float:
        """Function to calculate the reward for a single RL agent.

        Arguments
        ---------
        agent: LocalHyperdriveAgent
            The agent to calculate the reward for.

        Returns
        -------
        float
            The reward for the agent.
        """
        raise NotImplementedError

    ######### Setup functions ########

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

        eval_mode = self.env_config.eval_mode
        if eval_mode:
            db_port = 5434
            chain_port = 10001
        else:
            db_port = 5435 + self.worker_index
            chain_port = 10002 + self.worker_index

        # Instantiate the random number generator
        random_seed = np.random.randint(1, 99999) + self.worker_index
        self.rng = np.random.default_rng(random_seed)

        # Set up chain
        local_chain_config = LocalChain.Config(
            block_timestamp_interval=12,
            db_port=db_port,
            chain_port=chain_port,
            calc_pnl=eval_mode,
            manual_database_sync=(not eval_mode),
            backfill_pool_info=False,
        )

        self.chain = LocalChain(local_chain_config)

        # Define agents
        assert self.env_config.num_agents > 0
        self._agent_ids = {f"{AGENT_PREFIX}{i}" for i in range(self.env_config.num_agents)}
        # TODO using agent0's agents for now, which is tied to hyperdrive. Ideally
        # these agents would be general purpose.
        self.agents = {
            agent_id: self.chain.init_agent(
                eth=FixedPoint(100),
                name=agent_id,
            )
            for agent_id in self._agent_ids
        }

        # Set up member variables potentially needed by subclasses
        self.step_count = 0
        # setup logger
        self.logger = logging.getLogger()
        self.logger.info("rng seed: " + str(random_seed))

        # Call any initialization setup before we snapshot
        self.setup_environment()

        # Save a snapshot of initial conditions for resets
        self.chain.save_snapshot()

        # Set necessary member variables needed by the underlying environment class
        assert self.env_config.render_mode is None or self.env_config.render_mode in self.metadata["render_modes"]
        self.render_mode = self.env_config.render_mode
        # Set up action and observation space
        self.action_space = spaces.Dict({agent_id: self.create_action_space() for agent_id in self._agent_ids})
        self.observation_space = self._create_observation_space()
        self._action_space_in_preferred_format = True
        self._obs_space_in_preferred_format = True

        # episode variables
        self._prev_pnls: dict[str, float] = {agent_id: 0.0 for agent_id in self._agent_ids}

        super().__init__()

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
        self.step_count = 0
        self._terminateds = set()
        self._truncateds = set()

        # Get first observation and info
        observations = self._get_observations()
        info = self._get_info()

        return observations, info

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
        self.logger.info(f"\nStep {self.step_count} Time: {datetime.now().strftime('%I:%M:%S %p')}")

        self._apply_action(action_dict=action_dict)

        # We minimize time between bot making an action, so we advance time after actions have been made
        # but before the observation
        self.chain.advance_time(self.env_config.step_advance_time, create_checkpoints=True)

        self.step_environment()

        observations = self._get_observations(agent_ids=action_dict.keys())
        info = self._get_info(agent_ids=action_dict.keys())

        step_rewards = self._calculate_rewards(agent_ids=action_dict.keys())

        episode_over = self.step_count >= self.env_config.episode_length - 1

        truncateds = {agent_id: episode_over for agent_id in action_dict.keys()}
        terminateds = {agent_id: False for agent_id in action_dict.keys()}

        self._truncateds.update({agent_id for agent_id, truncated in truncateds.items() if truncated})
        self._terminateds.update({agent_id for agent_id, terminated in terminateds.items() if terminated})

        truncateds["__all__"] = len(self._truncateds) == len(self._agent_ids)
        terminateds["__all__"] = len(self._terminateds) == len(self._agent_ids)

        self.step_count += 1
        # TODO when does the episode stop?
        return observations, step_rewards, terminateds, truncateds, info

    def _create_observation_space(self) -> spaces.Dict:
        """Function to get observations from the environment, and create the observation space."""
        observation = self._get_observations()
        obs_shape = None
        out_obs_space = {}
        for agent_id, obs in observation.items():
            curr_obs_shape = obs.shape
            assert len(curr_obs_shape) == 1, "Observation space must be 1D"
            if obs_shape is None:
                obs_shape = curr_obs_shape
            else:
                assert obs_shape == curr_obs_shape, "Observation space must be the same for all agents"

            out_obs_space[agent_id] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=obs_shape,
                dtype=np.float64,
            )
        return spaces.Dict(out_obs_space)

    def _get_observations(self, agent_ids: Iterable[str] | None = None) -> dict[str, np.ndarray]:
        """Function to get observations from the environment.

        This function calls `get_shared_observation` and `get_agent_observation`,
        and concatenates them into a single observation for each agent
        """
        # Agents passed in means it might be a subset of agents being called.
        if agent_ids is None:
            agents = self.agents
        else:
            agents = {agent_id: self.agents[agent_id] for agent_id in agent_ids}

        shared_observations = self.get_shared_observation()
        out_obs = {}
        for agent_id in agents:
            agent_observations = self.get_agent_observation(agents[agent_id])
            out_obs[agent_id] = np.concatenate(
                [shared_observations, agent_observations],
            )
        return out_obs

    def _get_info(self, agent_ids: Iterable[str] | None = None) -> dict[str, Any]:
        if agent_ids is None:
            agents = self.agents
        else:
            agents = {agent_id: self.agents[agent_id] for agent_id in agent_ids}
        # TODO expose get_info to subclasses
        info_dict = {agent_id: {} for agent_id in agents}
        return info_dict

    def _apply_action(self, action_dict: dict[str, np.ndarray]) -> None:
        # Agents passed in means it might be a subset of agents being called.
        for agent_id, action in action_dict.items():
            _ = self.apply_action(self.agents[agent_id], action)

    def _calculate_rewards(self, agent_ids: Iterable[str] | None = None) -> dict[str, float]:
        # Agents passed in means it might be a subset of agents being called.
        if agent_ids is None:
            agents = self.agents
        else:
            agents = {agent_id: self.agents[agent_id] for agent_id in agent_ids}

        out_rewards = {}
        for agent_id, agent in agents.items():
            out_rewards[agent_id] = self.calculate_agent_reward(agent)
        return out_rewards

    def render(self) -> None:
        """Renders the environment. No rendering available for hyperdrive env."""
        return None
