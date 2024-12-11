"""A hyperdrive rl gym environment."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Type

import numpy as np
import pandas as pd
from agent0 import LocalHyperdrive
from fixedpointmath import FixedPoint
from gymnasium import spaces
from scipy.special import expit

from .base_env import BaseEnv
from .rewards import DeltaPnl
from .variable_rate_policy import RandomNormalVariableRate, VariableRatePolicy

if TYPE_CHECKING:
    from agent0.core.hyperdrive.interactive.local_hyperdrive_agent import LocalHyperdriveAgent

    from .rewards.base_reward import BaseReward

AGENT_PREFIX = "agent"
POLICY_PREFIX = "policy"

# Global suppression of warnings, TODO fix
warnings.filterwarnings("ignore")


class TradeTypes(Enum):
    """Enum denoting between long and short indices"""

    LONG = 0
    SHORT = 1


class RayHyperdriveEnv(BaseEnv):
    """A simple hyperdrive environment that allows for 2 positions, long and short."""

    # pylint: disable=too-many-instance-attributes

    @dataclass(kw_only=True)
    class Config(BaseEnv.Config):
        """The configuration for RayHyperdriveEnv."""

        # Reward and variable rate policy
        # Note both of these policies are specific to hyperdrive
        variable_rate_policy: VariableRatePolicy = field(default=RandomNormalVariableRate())
        reward_policy: Type[BaseReward] = field(default=DeltaPnl)

        # Defines which columns from pool config to include in the observation space
        pool_config_columns: list[str] = field(
            default_factory=lambda: [
                "initial_vault_share_price",
                "minimum_share_reserves",
                "minimum_transaction_amount",
                "circuit_breaker_delta",
                "position_duration",
                "checkpoint_duration",
                "time_stretch",
                "curve_fee",
                "flat_fee",
                "governance_lp_fee",
                "governance_zombie_fee",
                "inv_time_stretch",
            ]
        )

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

    env_config: RayHyperdriveEnv.Config

    def init_config(self, env_config) -> None:
        if env_config.get("env_config") is None:
            self.env_config = self.Config()
        else:
            # We check the type after in an assertion
            self.env_config = env_config["env_config"]  # type: ignore
            assert isinstance(self.env_config, RayHyperdriveEnv.Config)

        # Set env shortcuts needed by other functions here
        self.eval_mode = self.env_config.eval_mode
        self.sample_actions = self.env_config.sample_actions

    def setup_environment(self):
        initial_pool_config = self._get_hyperdrive_pool_config()
        self.interactive_hyperdrive = LocalHyperdrive(self.chain, initial_pool_config)

        if self.eval_mode and self.worker_index == 0:
            self.chain.run_dashboard()

        for agent in self.agents.values():
            agent.set_active(pool=self.interactive_hyperdrive)
            agent.add_funds(base=self.env_config.rl_agent_budget)

        self.interactive_hyperdrive.sync_database()

        # Setup the reward policy
        self.reward = self.env_config.reward_policy(env=self)

    def reset_env(self):
        self.interactive_hyperdrive.sync_database()
        # Call reset on variable rate policy
        self.env_config.variable_rate_policy.reset(self.rng)
        # TODO do we also need to reset the reward policy?

    def create_action_space(self) -> spaces.Box:
        """Function to create the action space for a single agent in the environment.

        TODO there may be things we can abstract out here for a general purpose
        trading environment.

        TODO define this space in a dictionary format for readability,
        and flatten in base environment.
        """

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

        # Tracker for action lengths
        # first TradeTypes * `max_positions_per_type` is for closing existing positions
        # the + TradeTypes * 2 is for opening a new trade (2 indicates probability & volume)
        # the +4 is for LP
        # (longs, shorts) -> close_order_i(logit), new_order(logit), volume)
        # (lp) -> add_lp_order(logit), volume_add_lp, remove_lp_order(logit), volume_remove_lp)

        self.action_length_per_trade_set = len(TradeTypes) * (self.env_config.max_positions_per_type + 2) + 4
        return spaces.Box(
            low=-1e2,
            high=1e2,
            dtype=np.float64,
            shape=(self.action_length_per_trade_set,),
        )

    def _get_hyperdrive_pool_config(self) -> LocalHyperdrive.Config:
        """Get the Hyperdrive pool config."""
        return LocalHyperdrive.Config()

    def apply_action(self, agent: LocalHyperdriveAgent, action: np.ndarray) -> bool:
        # TODO
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-nested-blocks
        # pylint: disable=too-many-statements
        # Length is 2 * len(TradeTypes) for open/close long/short + 1 for LP
        trade_success = [
            True,
        ] * (2 * len(TradeTypes) + 1)

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
        agent_positions = agent.get_positions(coerce_float=False)

        # Closing trades
        close_trade_success = self._apply_close_trades(agent, close_long_short_actions, agent_positions)
        trade_success[: len(close_trade_success)] = close_trade_success
        # Open trades
        open_trade_success = self._apply_open_trades(agent, min_tx_amount, open_long_short_actions, agent_positions)
        trade_success[len(close_trade_success) : len(open_trade_success)] = open_trade_success
        # LP trade
        trade_success[-1] = self._apply_lp_trades(agent, min_tx_amount, lp_actions)

        return all(trade_success)

    def _apply_lp_trades(self, agent: LocalHyperdriveAgent, min_tx_amount: FixedPoint, lp_actions: np.ndarray) -> bool:
        """Apply the LP trades."""
        trade_success = True
        agent_wallet = agent.get_wallet()
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
                agent.add_liquidity(add_lp_volume)
            if remove_lp and remove_lp_volume <= agent_wallet.lp_tokens:
                agent.remove_liquidity(remove_lp_volume)
            # Always try and remove withdrawal shares
            if agent_wallet.withdraw_shares > 0:
                # TODO error handling or check when withdrawal shares are not withdrawable
                agent.redeem_withdrawal_shares(agent_wallet.withdraw_shares)
        except Exception as err:  # pylint: disable=broad-except
            self.logger.warning(f"Failed to LP: {repr(err)}")
            trade_success = False
        return trade_success

    def _apply_close_trades(
        self, agent: LocalHyperdriveAgent, close_long_short_actions: np.ndarray, agent_positions: pd.DataFrame
    ) -> list[bool]:
        """Close trades."""
        trade_success = [
            True,
        ] * len(TradeTypes)
        for i, trade_type in enumerate(TradeTypes):
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
                        agent.close_long(
                            maturity_time=int(position_to_close["maturity_time"]),
                            bonds=FixedPoint(position_to_close["token_balance"]),
                        )
                    elif trade_type == TradeTypes.SHORT:
                        agent.close_short(
                            maturity_time=int(position_to_close["maturity_time"]),
                            bonds=FixedPoint(position_to_close["token_balance"]),
                        )
            except Exception as err:  # pylint: disable=broad-except
                self.logger.warning(f"Failed to close {trade_type} trade: {repr(err)}")
                trade_success[i] = False
        return trade_success

    def _apply_open_trades(
        self,
        agent: LocalHyperdriveAgent,
        min_tx_amount: FixedPoint,
        open_long_short_actions: np.ndarray,
        agent_positions: pd.DataFrame,
    ) -> list[bool]:
        """Apply open trades."""
        trade_success = [
            True,
        ] * len(TradeTypes)
        for i, trade_type in enumerate(TradeTypes):
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
                        agent_wallet_balance = agent.get_wallet().balance.amount
                        if trade_type == TradeTypes.LONG and agent_wallet_balance >= min_tx_amount:
                            # Get the agent's max trade amount
                            max_long_amount = self.interactive_hyperdrive.interface.calc_max_long(
                                budget=agent_wallet_balance
                            )
                            # While volume isn't strictly a probability, we interpret it as a value between 0 and 1
                            amount_probability = FixedPoint(expit(open_long_short_actions[trade_type.value, 1]))
                            # Map the probability to be between the min and max transaction amounts
                            volume_adjusted = min_tx_amount + amount_probability * (max_long_amount - min_tx_amount)
                            agent.open_long(base=volume_adjusted)
                        elif trade_type == TradeTypes.SHORT and agent_wallet_balance >= min_tx_amount:
                            # Get the agent's max trade amount
                            agent_wallet_balance = agent.get_wallet().balance.amount
                            max_short_amount = self.interactive_hyperdrive.interface.calc_max_short(
                                budget=agent_wallet_balance
                            )
                            # While volume isn't strictly a probability, we interpret it as a value between 0 and 1
                            amount_probability = FixedPoint(expit(open_long_short_actions[trade_type.value, 1]))
                            # Map the probability to be between the min and max transaction amounts
                            volume_adjusted = min_tx_amount + amount_probability * (max_short_amount - min_tx_amount)
                            agent.open_short(bonds=volume_adjusted)
                    # Base exception here to catch rust errors
                    except BaseException as err:  # pylint: disable=broad-except
                        self.logger.warning(f"Failed to open {trade_type} trade: {repr(err)}")
                        trade_success[i] = False
        return trade_success

    def step_environment(self):
        # Update variable rate with probability Config.rate_change_probability
        # TODO: Parameterize distribution and clip
        if self.env_config.variable_rate_policy.do_change_rate(self.rng):
            new_rate = self.env_config.variable_rate_policy.get_new_rate(
                self.interactive_hyperdrive.interface, self.rng
            )
            self.interactive_hyperdrive.set_variable_rate(new_rate)

    def get_shared_observations(self) -> np.ndarray:
        self.interactive_hyperdrive.sync_database()
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
        return np.concatenate(
            [
                pool_config,
                pool_info,
                block_timestamp,
                steps_remaining,
            ]
        )

    def get_agent_observations(self, agent: LocalHyperdriveAgent) -> np.ndarray:

        # Long and short features: token balance, pnl, time to maturity
        num_long_features = self.env_config.max_positions_per_type * 3
        num_short_features = self.env_config.max_positions_per_type * 3
        # LP features: token balance, pnl
        num_lp_features = 2
        num_wallet_features = 1

        # Long Features: trade type, order_i -> [volume, value, normalized_time_remaining]
        long_features = np.zeros(num_long_features, dtype=np.float64)
        # Short Features: trade type, order_i -> [volume, value, normalized_time_remaining]
        short_features = np.zeros(num_short_features, dtype=np.float64)
        # LP: -> [volume, value]
        lp_features = np.zeros(num_lp_features, dtype=np.float64)
        # Additional features: pnl
        wallet_features = np.zeros(num_wallet_features, dtype=np.float64)

        current_block = self.interactive_hyperdrive.interface.get_current_block()
        timestamp = self.interactive_hyperdrive.interface.get_block_timestamp(current_block)

        # Observation data uses floats
        open_agent_positions = agent.get_positions(coerce_float=True, calc_pnl=True)
        all_agent_positions = agent.get_positions(coerce_float=True, calc_pnl=True, show_closed_positions=True)

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

        return np.concatenate(
            [
                long_features,
                short_features,
                lp_features,
                wallet_features,
            ]
        )

    def _calculate_rewards(self, agent_ids: Iterable[str] | None = None) -> dict[str, float]:
        # Note since our reward calculation handles iterating over a collection of agents,
        # we overwrite the outer function to do this for us instead of overwriting
        # `calculate_agent_reward`.
        return self.reward.calculate_rewards(agent_ids)
