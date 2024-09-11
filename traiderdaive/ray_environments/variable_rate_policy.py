"""Defining policy classes for controlling variable rates."""

from dataclasses import dataclass
from typing import Sequence, Type

import numpy as np
from agent0.ethpy.hyperdrive import HyperdriveReadInterface
from fixedpointmath import FixedPoint
from numpy.random import Generator


class VariableRatePolicy:
    @dataclass(kw_only=True)
    class Config:
        rate_change_probability: float = 0.1

    def __init__(self, config: Config | None = None, rng: Generator | None = None) -> None:
        if config is None:
            config = self.Config()
        self.config = config
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

    def do_change_rate(self) -> bool:
        """Function defining when to change the rate.

        This function gets called every `step`, and returns `True` if the rate should be changed.
        The default behavior is to change the rate based on `rate_change_probability`.

        Returns
        -------
        bool
            Whether or not to change the rate
        """
        # Type narrowing
        assert self.rng is not None
        if self.rng.random() < self.config.rate_change_probability:
            return True
        return False

    def get_new_rate(self, interface: HyperdriveReadInterface) -> FixedPoint:
        """Function defining behavior of how to change the variable rate.

        Arguments
        ---------
        interface: HyperdriveReadInterface
            The interface to the hyperdrive.

        Returns
        -------
        FixedPoint
            The new variable rate to set to.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Resets the policy.

        This function gets called whenever the environment's `reset` is called.
        """
        # Default behavior is no-op


class RandomNormalVariableRate(VariableRatePolicy):
    @dataclass(kw_only=True)
    class Config(VariableRatePolicy.Config):
        loc: float = 1.0
        scale: float = 0.01

    def get_new_rate(self, interface: HyperdriveReadInterface) -> FixedPoint:
        # Type narrow
        assert isinstance(self.config, RandomNormalVariableRate.Config)
        assert self.rng is not None
        current_rate = interface.get_variable_rate()
        # narrow type
        assert current_rate is not None
        # new rate is random & between 10x and 0.1x the current rate
        return current_rate * FixedPoint(
            np.minimum(10.0, np.maximum(0.1, self.rng.normal(loc=self.config.loc, scale=self.config.scale)))
        )


class RandomWalk(RandomNormalVariableRate):
    @dataclass(kw_only=True)
    class Config(RandomNormalVariableRate.Config):
        rate_change_probability: float = 0.1
        loc: float = 1.0
        scale: float = 0.01


class Transition(RandomNormalVariableRate):
    @dataclass(kw_only=True)
    class Config(RandomNormalVariableRate.Config):
        rate_change_probability: float = 0.01
        loc: float = 1.0
        scale: float = 1.0


class Swings(RandomNormalVariableRate):
    @dataclass(kw_only=True)
    class Config(RandomNormalVariableRate.Config):
        rate_change_probability: float = 0.1
        loc: float = 1.0
        scale: float = 0.1


class Ramp(VariableRatePolicy):
    @dataclass(kw_only=True)
    class Config(VariableRatePolicy.Config):
        # We always change the rate on every step
        rate_change_probability = 1
        rate_change_delta = FixedPoint(0.001)

    def get_new_rate(self, interface: HyperdriveReadInterface) -> FixedPoint:
        # Type narrow
        assert isinstance(self.config, Ramp.Config)
        assert self.rng is not None
        current_rate = interface.get_variable_rate()
        # narrow type
        assert current_rate is not None
        # new rate is random & between 10x and 0.1x the current rate
        return current_rate + self.config.rate_change_delta


class PositiveRamp(Ramp):
    @dataclass(kw_only=True)
    class Config(Ramp.Config):
        rate_change_probability: float = 0.1
        rate_change_delta = FixedPoint(0.001)


class NegativeRamp(Ramp):
    @dataclass(kw_only=True)
    class Config(Ramp.Config):
        rate_change_probability: float = 0.1
        rate_change_delta = FixedPoint(-0.001)


class RandomRatePolicy(VariableRatePolicy):
    @dataclass(kw_only=True)
    class Config(VariableRatePolicy.Config):
        policies: Sequence[Type[VariableRatePolicy]] = (
            RandomWalk,
            Transition,
            Swings,
            PositiveRamp,
            NegativeRamp,
        )

    def __init__(self, config: Config | None = None, rng: Generator | None = None) -> None:
        super().__init__(config, rng)
        self.active_policy: VariableRatePolicy | None = None

    def reset(self):
        # In reset, we select a random policy to use
        assert self.rng is not None
        assert isinstance(self.config, RandomRatePolicy.Config)
        # rng.choice has issues mapping list of types to array like
        active_policy_class: Type[VariableRatePolicy] = self.rng.choice(self.config.policies)  # type: ignore
        # TODO: Allow user to specify configs for each option in the policies list
        self.active_policy = active_policy_class(active_policy_class.Config(), self.rng)
        print(f"Using {self.active_policy}")

    def do_change_rate(self) -> bool:
        assert self.active_policy is not None
        return self.active_policy.do_change_rate()

    def get_new_rate(self, interface: HyperdriveReadInterface) -> FixedPoint:
        assert self.active_policy is not None
        return self.active_policy.get_new_rate(interface)
