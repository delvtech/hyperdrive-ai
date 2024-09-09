"""A hyperdrive rl gym environment."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Iterable

import numpy as np
from agent0 import LocalChain, LocalHyperdrive, PolicyZoo
from fixedpointmath import FixedPoint
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from scipy.special import expit

AGENT_PREFIX = "agent"
POLICY_PREFIX = "policy"

# Global suppression of warnings, TODO fix
warnings.filterwarnings("ignore")


class TradeTypes(Enum):
    """Enum denoting between long and short indices"""

    LONG = 0
    SHORT = 1


class RayHyperdriveEnv(MultiAgentEnv):
    """A simple hyperdrive environment that allows for 2 positions, long and short."""

    # pylint: disable=too-many-instance-attributes

    @dataclass(kw_only=True)
    class Config:
        """The configuration for RayHyperdriveEnv."""

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

        # Hyperdrive Config
        # How much to advance time per step
        step_advance_time = 8 * 3600  # 8 hours
        # Probability of updating the variable rate
        rate_change_probability: float = 0.1

        # RL Agents Config
        num_agents: int = 4
        # The constant trade amounts for longs and shorts
        rl_agent_budget: FixedPoint = FixedPoint(1_000_000)
        max_positions_per_type: int = 10
        base_reward_scale: float = 0.0
        # The threshold for the probability of opening and closing orders
        open_threshold: float = 0.5
        close_threshold: float = 0.5

        # Other bots config
        num_random_bots: int = 0
        num_random_hold_bots: int = 0
        random_bot_budget: FixedPoint = FixedPoint(1_000_000)

        # TODO: Check if PPO is already sampling actions!
        sample_actions: bool = False
        # Sets alternate ports for eval to avoid connecting to a training chain
        eval_mode: bool = False

        # Defines which columns from pool info to include in the observation space
        pool_info_columns: list[str] = field(
            default_factory=lambda: [
                "epoch_timestamp",
                "share_reserves",
                "share_adjustment",
                "zombie_base_proceeds",
                "zombie_share_reserves",
                "bond_reserves",
                "lp_total_supply",
                "vault_share_price",
                "longs_outstanding",
                "long_average_maturity_time",
                "shorts_outstanding",
                "short_average_maturity_time",
                "withdrawal_shares_ready_to_withdraw",
                "withdrawal_shares_proceeds",
                "lp_share_price",
                "long_exposure",
                "total_supply_withdrawal_shares",
                "gov_fees_accrued",
                "hyperdrive_base_balance",
                "hyperdrive_eth_balance",
                "variable_rate",
                "vault_shares",
                "spot_price",
                "fixed_rate",
            ]
        )

    # Defines allowed render modes and fps
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        env_config,
    ):
        """Initializes the environment"""
        self.worker_index = env_config.worker_index

        if env_config.get("env_config") is None:
            self.env_config = self.Config()
        else:
            self.env_config = env_config["env_config"]
        # TODO parameterize these in the gym config

        # Multiagent setup
        self.agents = {f"{AGENT_PREFIX}{i}" for i in range(self.env_config.num_agents)}
        self._agent_ids = self.agents
        self.terminateds = set()
        self.truncateds = set()

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

        initial_pool_config = LocalHyperdrive.Config()
        self.chain = LocalChain(local_chain_config)
        self.interactive_hyperdrive = LocalHyperdrive(self.chain, initial_pool_config)

        if self.eval_mode:
            self.chain.run_dashboard()

        # TODO set seed
        self.rng = np.random.default_rng()

        # Define the rl agents
        self.rl_agents = {
            name: self.chain.init_agent(
                base=self.env_config.rl_agent_budget,
                eth=FixedPoint(100),
                pool=self.interactive_hyperdrive,
                name=name,
            )
            for name in self.agents
        }

        # Define the random bots
        self.random_bots = [
            self.chain.init_agent(
                base=self.env_config.random_bot_budget,
                eth=FixedPoint(100),
                pool=self.interactive_hyperdrive,
                policy=PolicyZoo.random,
                # TODO set the seed per random bot here for reproducibility
                # TODO omitting rng_seed results in the same random generators
                # for all bots, fix
                policy_config=PolicyZoo.random.Config(rng_seed=i),
                name="random_bot_" + str(i),
            )
            for i in range(self.env_config.num_random_bots)
        ]

        self.random_bots.extend(
            [
                self.chain.init_agent(
                    base=self.env_config.random_bot_budget,
                    eth=FixedPoint(100),
                    pool=self.interactive_hyperdrive,
                    policy=PolicyZoo.random_hold,
                    # TODO set the seed per random bot here for reproducibility
                    policy_config=PolicyZoo.random_hold.Config(
                        trade_chance=FixedPoint("0.8"),
                        max_open_positions=1000,
                        # TODO omitting rng_seed results in the same random generators
                        # for all bots, fix
                        rng_seed=self.env_config.num_random_bots + i,
                    ),
                    name="random_hold_bot_" + str(i),
                )
                for i in range(self.env_config.num_random_hold_bots)
            ]
        )

        self.interactive_hyperdrive.sync_database()

        # Save a snapshot of initial conditions for resets
        self.chain.save_snapshot()

        assert self.env_config.render_mode is None or self.env_config.render_mode in self.metadata["render_modes"]
        self.render_mode = self.env_config.render_mode

        # The space of allowed actions to take
        # Following https://github.com/AminHP/gym-mtsim
        # These actions are encoded into a 1d vector of continuous values
        # This is due to not all algorithms supporting dict or multidimention box actions

        # Here, these actions are for 2 types of trades: longs, shorts
        # each encoded as an array of length max_positions + 2
        # For a given type of trade, the elements are interpreted as
        # [
        #    probability of closing order 1,
        #    probability of closing order 2,
        #    ...
        #    probability of closing order max_positions,
        #    probability of holding or creating a new order,
        #    volume of the new order
        # ]
        # The last two define the probability of creating a new order (or no op), with the volume of the new order
        # Probabilities are in logit space to ensure probability values are in range [0, 1]

        # The final 4 fields specify LP positions, interpreted as
        # [
        #    probability of adding liquidity,
        #    add liquidity volume,
        #    probability of removing liquidity,
        #    add liquidity volume,
        # ]

        # (longs, shorts) -> close_order_i(logit), new_order(logit), volume)
        # (lp) -> add_lp_order(logit), volume_add_lp, remove_lp_order(logit), volume_remove_lp)
        self._action_space_in_preferred_format = True
        self.action_space = spaces.Dict(
            {
                agent_id: spaces.Box(
                    low=-1e2,
                    high=1e2,
                    dtype=np.float64,
                    shape=(len(TradeTypes) * (self.env_config.max_positions_per_type + 2) + 4,),
                )
                for agent_id in self.agents
            }
        )

        # Observation space is
        # TODO add more features
        # Pool Features: spot price, lp share price
        # TODO use pnl instead of value
        # TODO add bookkeeping for entry spot price
        # Long Orders: trade type, order_i -> [volume, value, normalized_time_remaining]
        # Short Orders: trade type, order_i -> [volume, value, normalized_time_remaining]
        # LP: -> [volume, value]
        # Here, orders_i is a direct mapping to agent.wallet
        # Note normalize_time_to_maturity will always be 0 for LP positions
        self.num_pool_features = len(self.env_config.pool_info_columns)
        # Long and short features: token balance, pnl, time to maturity
        self.num_long_features = self.env_config.max_positions_per_type * 3
        self.num_short_features = self.env_config.max_positions_per_type * 3
        # LP features: token balance, pnl
        self.num_lp_features = 2
        inf = 1e10
        self._obs_space_in_preferred_format = True
        # OS shape = # pool features + max positions per type x
        self.observation_space_shape = (
            self.num_pool_features + self.num_long_features + self.num_short_features + self.num_lp_features,
        )
        self.observation_space = spaces.Dict(
            {
                agent_id: spaces.Box(
                    low=-inf,
                    high=inf,
                    shape=self.observation_space_shape,
                    dtype=np.float64,
                )
                for agent_id in self.agents
            }
        )

        # episode variables
        self._prev_pnls: dict[str, float] = {agent_id: 0.0 for agent_id in self.agents}
        self._step_count = 0

        self.logger = logging.getLogger()
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

        self.interactive_hyperdrive.sync_database()

        # Reset internal member variables
        self._prev_pnls: dict[str, float] = {agent_id: 0.0 for agent_id in self.agents}
        self._step_count = 0
        self.terminateds = set()
        self.truncateds = set()

        # Get first observation and info
        observations = self._get_observations()
        info = self._get_info()

        return observations, info

    def _apply_action(self, agent_id: str, action: np.ndarray) -> bool:
        """Execute the bot action on-chain.

        Arguments
        ---------
        agent_id: str
            Unique identifying string for the agent.
        action: np.ndarray
            Action activations returned by the policy network.

        Returns
        -------
        bool
            True if the trade was successful, False otherwise.
        """
        # TODO
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-nested-blocks
        # pylint: disable=too-many-statements
        trade_success = True

        # The actual min txn amount is a function of pool state. Without helper functions, we simply add a safe amount.
        min_tx_amount = self.interactive_hyperdrive.config.minimum_transaction_amount * FixedPoint("2")

        long_short_actions = action[:-4]
        long_short_actions = long_short_actions.reshape((len(TradeTypes), self.env_config.max_positions_per_type + 2))
        close_long_short_actions = long_short_actions[:, :-2]
        open_long_short_actions = long_short_actions[:, -2:]
        lp_actions = action[-4:]

        # TODO should likely try and handle these trades as fast as possible, or eventually
        # allow for reordering.
        # Current solution is to minimize the amount of time between trades within a step
        # and accelerate time in a single step at the end of a step.

        # TODO: Add 6 additional logit dimensions that indicate trade order.
        # Then we can sort by those and the agent can specify the order of trades.
        # The RL bot handles trades in this order:
        # (1) Close long tokens
        # (2) Close short tokens
        # (2) Open long tokens
        # (4) Open short tokens
        # (5) Add liquidity
        # (6) Remove liquidity
        # (7) Redeem withdrawal shares

        # Get agent positions once so we don't hit the database too many times.
        # The downside of this is we could undershoot the max_positions_per_type
        # since any close actions this round will not be accounted for. This
        # is a fine tradeoff, though, since it's an undershoot and the next time
        # apply_action is called the previous closes will be accounted for.
        agent_positions = self.rl_agents[agent_id].get_positions(coerce_float=False)

        # Closing trades
        for trade_type in TradeTypes:
            # Get agent positions for this trade type
            trade_positions = agent_positions[agent_positions["token_type"] == trade_type.name]

            # Ensure positions are sorted from oldest to newest. The action
            # space is sorted this way. The agent itself is going to choose an
            # ordering.
            trade_positions = trade_positions.sort_values("maturity_time")
            num_trade_positions = len(trade_positions)

            # Handle closing orders
            # The index of orders here is from oldest to newest
            # TODO (sheng) if we want the rl bot to explicitly learn how to
            # close orders based on the orders input feature, we can shuffle the
            # order of closing orders and match them here in this case we would
            # need to also shuffle the obs space in the exact same way.
            close_orders_probability = expit(close_long_short_actions[trade_type.value, :])
            if self.sample_actions:
                random_roll = self.rng.uniform(0, 1, len(close_orders_probability))
                orders_to_close_index = np.nonzero(random_roll <= close_orders_probability)[0]
            else:
                orders_to_close_index = np.nonzero(close_orders_probability > self.env_config.close_threshold)[0]

            # Filter orders to close to be only the number of trade positions
            orders_to_close_index = orders_to_close_index[orders_to_close_index < num_trade_positions]
            positions_to_close = trade_positions.iloc[orders_to_close_index]

            # Close positions
            try:
                for _, position_to_close in positions_to_close.iterrows():
                    if trade_type == TradeTypes.LONG:
                        self.rl_agents[agent_id].close_long(
                            maturity_time=int(position_to_close["maturity_time"]),
                            bonds=FixedPoint(position_to_close["token_balance"]),
                        )
                    elif trade_type == TradeTypes.SHORT:
                        self.rl_agents[agent_id].close_short(
                            maturity_time=int(position_to_close["maturity_time"]),
                            bonds=FixedPoint(position_to_close["token_balance"]),
                        )
            except Exception as err:  # pylint: disable=broad-except
                self.logger.warning(f"Failed to close {trade_type} trade: {repr(err)}")
                trade_success = False

        # Open trades
        for trade_type in TradeTypes:
            # Get agent positions again after closing
            trade_positions = agent_positions[agent_positions["token_type"] == trade_type.name]
            num_trade_positions = len(trade_positions)
            # Only open trades if we haven't maxed out positions
            if num_trade_positions < self.env_config.max_positions_per_type:
                new_order_probability = expit(open_long_short_actions[trade_type.value, 0])
                # Opening orders
                if self.sample_actions:
                    open_order = self.rng.uniform(0, 1) <= new_order_probability
                else:
                    open_order = new_order_probability > self.env_config.open_threshold

                if open_order:
                    try:
                        # Need to get wallet inside this loop since each loop
                        # iteration could include an open that reduces balance.
                        agent_wallet_balance = self.rl_agents[agent_id].get_wallet().balance.amount
                        if trade_type == TradeTypes.LONG and agent_wallet_balance >= min_tx_amount:
                            # Get the agent's max trade amount
                            max_long_amount = self.interactive_hyperdrive.interface.calc_max_long(
                                budget=agent_wallet_balance
                            )
                            # While volume isn't strictly a probability, we interpret it as a value between 0 and 1
                            amount_probability = FixedPoint(expit(open_long_short_actions[trade_type.value, 1]))
                            # Map the probability to be between the min and max transaction amounts
                            volume_adjusted = min_tx_amount + amount_probability * (max_long_amount - min_tx_amount)
                            self.rl_agents[agent_id].open_long(base=volume_adjusted)
                        elif trade_type == TradeTypes.SHORT and agent_wallet_balance >= min_tx_amount:
                            # Get the agent's max trade amount
                            agent_wallet_balance = self.rl_agents[agent_id].get_wallet().balance.amount
                            max_short_amount = self.interactive_hyperdrive.interface.calc_max_short(
                                budget=agent_wallet_balance
                            )
                            # While volume isn't strictly a probability, we interpret it as a value between 0 and 1
                            amount_probability = FixedPoint(expit(open_long_short_actions[trade_type.value, 1]))
                            # Map the probability to be between the min and max transaction amounts
                            volume_adjusted = min_tx_amount + amount_probability * (max_short_amount - min_tx_amount)
                            self.rl_agents[agent_id].open_short(bonds=volume_adjusted)
                    # Base exception here to catch rust errors
                    except BaseException as err:  # pylint: disable=broad-except
                        self.logger.warning(f"Failed to open {trade_type} trade: {repr(err)}")
                        trade_success = False

        # LP actions
        agent_wallet = self.rl_agents[agent_id].get_wallet()
        lp_actions_expit = expit(lp_actions)
        if agent_wallet.balance.amount >= min_tx_amount:
            add_lp_probability = lp_actions_expit[0]
            add_lp_volume = min_tx_amount + FixedPoint(lp_actions_expit[1]) * (
                agent_wallet.balance.amount - min_tx_amount
            )
        else:
            add_lp_probability = np.float64(0)
            add_lp_volume = FixedPoint(0)

        if agent_wallet.lp_tokens >= min_tx_amount:
            remove_lp_probability = lp_actions_expit[2]
            remove_lp_volume = min_tx_amount + FixedPoint(lp_actions_expit[3]) * (
                agent_wallet.lp_tokens - min_tx_amount
            )
        else:
            remove_lp_probability = np.float64(0)
            remove_lp_volume = FixedPoint(0)

        if self.sample_actions:
            random_roll = self.rng.uniform(0, 1, 2)
            add_lp = random_roll[0] <= add_lp_probability
            remove_lp = random_roll[1] <= remove_lp_probability
        else:
            add_lp = add_lp_probability > self.env_config.open_threshold
            remove_lp = remove_lp_probability > self.env_config.close_threshold

        try:
            if add_lp and add_lp_volume <= agent_wallet.balance.amount:
                self.rl_agents[agent_id].add_liquidity(add_lp_volume)
            if remove_lp and remove_lp_volume <= self.rl_agents[agent_id].get_wallet().lp_tokens:
                self.rl_agents[agent_id].remove_liquidity(remove_lp_volume)
            # Always try and remove withdrawal shares
            if self.rl_agents[agent_id].get_wallet().withdraw_shares > 0:
                # TODO error handling or check when withdrawal shares are not withdrawable
                self.rl_agents[agent_id].redeem_withdrawal_share(self.rl_agents[agent_id].get_wallet().withdraw_shares)
        except Exception as err:  # pylint: disable=broad-except
            self.logger.warning(f"Failed to LP: {repr(err)}")
            trade_success = False

        return trade_success

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
        if np.random.rand() < self.env_config.rate_change_probability:
            current_rate = self.interactive_hyperdrive.interface.get_variable_rate()
            # new rate is random & between 10x and 0.1x the current rate
            new_rate = current_rate * FixedPoint(
                np.minimum(10.0, np.maximum(0.1, np.random.normal(loc=1.0, scale=0.01)))
            )
            self.interactive_hyperdrive.set_variable_rate(new_rate)

        self.interactive_hyperdrive.sync_database()

        observations = self._get_observations(agents=action_dict.keys())
        info = self._get_info(agents=action_dict.keys())
        step_rewards = self._calculate_rewards(agents=action_dict.keys())

        episode_over = self._step_count >= self.env_config.episode_length - 1

        truncateds = {agent_id: episode_over for agent_id in action_dict.keys()}
        terminateds = {agent_id: False for agent_id in action_dict.keys()}

        self.truncateds.update({agent_id for agent_id, truncated in truncateds.items() if truncated})
        self.terminateds.update({agent_id for agent_id, terminated in terminateds.items() if terminated})

        truncateds["__all__"] = len(self.truncateds) == len(self.agents)
        terminateds["__all__"] = len(self.terminateds) == len(self.agents)

        self._step_count += 1
        # TODO when does the episode stop?
        return observations, step_rewards, terminateds, truncateds, info

    def _get_info(self, agents: Iterable[str] | None = None) -> dict:
        agents = agents or self.agents
        info_dict = {agent_id: {} for agent_id in agents}
        return info_dict

    def _get_observations(self, agents: Iterable[str] | None = None) -> dict[str, np.ndarray]:
        # TODO: (dylan) If we're changing up pool config per episode (as we should be),
        # but it's constant within an episode, should we include it in the obs space?
        agents = agents or self.agents
        # Get the latest pool state feature from the db
        pool_state_df = self.interactive_hyperdrive.get_pool_info(coerce_float=True)
        pool_state_df = pool_state_df[self.env_config.pool_info_columns].iloc[-1].astype(float)
        pool_features = pool_state_df.values
        # TODO can also add other features, e.g., opening spot price

        out_obs = {}
        for agent_id in agents:
            # Long Features: trade type, order_i -> [volume, value, normalized_time_remaining]
            long_features = np.zeros(self.num_long_features)
            # Short Features: trade type, order_i -> [volume, value, normalized_time_remaining]
            short_features = np.zeros(self.num_short_features)
            # LP: -> [volume, value]
            lp_features = np.zeros(self.num_lp_features)

            # Observation data uses floats
            agent_positions = self.rl_agents[agent_id].get_positions(coerce_float=True, calc_pnl=True)

            if not agent_positions.empty:
                position_duration = self.interactive_hyperdrive.config.position_duration
                # We convert timestamp to epoch time here
                # We keep negative values for time past maturity
                current_block = self.interactive_hyperdrive.interface.get_current_block()
                timestamp = self.interactive_hyperdrive.interface.get_block_timestamp(current_block)
                agent_positions["normalized_time_remaining"] = (
                    agent_positions["maturity_time"] - timestamp
                ) / position_duration

                long_orders = agent_positions[agent_positions["token_type"] == "LONG"]
                # Ensure data is the same as the action space
                long_orders = long_orders.sort_values("maturity_time")
                long_orders = long_orders[["token_balance", "pnl", "normalized_time_remaining"]].values.flatten()

                short_orders = agent_positions[agent_positions["token_type"] == "SHORT"]
                # Ensure data is the same as the action space
                short_orders = short_orders.sort_values("maturity_time")
                short_orders = short_orders[["token_balance", "pnl", "normalized_time_remaining"]].values.flatten()

                lp_orders = agent_positions[agent_positions["token_type"] == "LP"]
                lp_orders = lp_orders[["token_balance", "pnl"]].values.flatten()

                # Add data to static size arrays
                long_features[: len(long_orders)] = long_orders
                short_features[: len(short_orders)] = short_orders
                lp_features[: len(lp_orders)] = lp_orders

            out_obs[agent_id] = np.concatenate(
                [
                    pool_features,
                    long_features,
                    short_features,
                    lp_features,
                ]
            )

        return out_obs

    def _calculate_rewards(self, agents: Iterable[str] | None = None) -> dict[str, float]:
        agents = agents or self.agents
        # The total delta for this episode

        current_positions = self.interactive_hyperdrive.get_positions(
            show_closed_positions=True, calc_pnl=True, coerce_float=True
        )
        reward = {}
        for agent_id in agents:
            # Filter by agent ID
            agent_positions = current_positions[current_positions["wallet_address"] == self.rl_agents[agent_id].address]
            # The agent_positions shows the pnl of all positions
            # Sum across all positions
            # TODO one option here is to only look at base positions instead of sum across all positions.
            # TODO handle the case where pnl calculation doesn't return a number
            # when you can't close the position

            total_pnl = float(agent_positions["pnl"].sum())

            # reward is in units of base
            # We use the change in pnl as the reward
            reward[agent_id] = total_pnl - self._prev_pnls[agent_id]
            self._prev_pnls[agent_id] = total_pnl

        return reward

    def render(self) -> None:
        """Renders the environment. No rendering available for hyperdrive env."""
        return None
