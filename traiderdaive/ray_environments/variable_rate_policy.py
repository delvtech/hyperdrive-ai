"""Defining policy classes for controlling variable rates."""

import logging
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

    def __init__(self, config: Config | None = None) -> None:
        if config is None:
            config = self.Config()
        self.config = config

    def do_change_rate(self, rng: Generator) -> bool:
        """Function defining when to change the rate.

        This function gets called every `step`, and returns `True` if the rate should be changed.
        The default behavior is to change the rate based on `rate_change_probability`.

        Arguments
        ---------
        rng: Generator
            The random number generator to use.

        Returns
        -------
        bool
            Whether or not to change the rate
        """
        # Type narrowing
        if rng.random() < self.config.rate_change_probability:
            return True
        return False

    def get_new_rate(self, interface: HyperdriveReadInterface, rng: Generator) -> FixedPoint:
        """Function defining behavior of how to change the variable rate.

        Arguments
        ---------
        interface: HyperdriveReadInterface
            The interface to the hyperdrive.
        rng: Generator
            The random number generator to use.

        Returns
        -------
        FixedPoint
            The new variable rate to set to.
        """
        raise NotImplementedError

    def reset(self, rng: Generator) -> None:
        """Resets the policy.

        This function gets called whenever the environment's `reset` is called.

        Arguments
        ---------
        rng: Generator
            The random number generator to use.
        """
        # Default behavior is no-op


class RandomNormalVariableRate(VariableRatePolicy):
    @dataclass(kw_only=True)
    class Config(VariableRatePolicy.Config):
        loc: float = 1.0
        scale: float = 0.01

    def get_new_rate(self, interface: HyperdriveReadInterface, rng: Generator) -> FixedPoint:
        # Type narrow
        assert isinstance(self.config, RandomNormalVariableRate.Config)
        current_rate = interface.get_variable_rate()
        # narrow type
        assert current_rate is not None
        # new rate is random & between 10x and 0.1x the current rate
        return current_rate * FixedPoint(
            np.minimum(10.0, np.maximum(0.1, rng.normal(loc=self.config.loc, scale=self.config.scale)))
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

    def get_new_rate(self, interface: HyperdriveReadInterface, rng: Generator) -> FixedPoint:
        # Type narrow
        assert isinstance(self.config, Ramp.Config)
        current_rate = interface.get_variable_rate()
        # narrow type
        assert current_rate is not None
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

    def __init__(self, config: Config | None = None):
        super().__init__(config)
        self.active_policy: VariableRatePolicy | None = None

    def reset(self, rng: Generator) -> None:
        # In reset, we select a random policy to use
        assert isinstance(self.config, RandomRatePolicy.Config)
        # rng.choice has issues mapping list of types to array like
        active_policy_class: Type[VariableRatePolicy] = rng.choice(self.config.policies)  # type: ignore
        self.active_policy = active_policy_class()
        logging.info(f"Using {self.active_policy}")

    def do_change_rate(self, rng: Generator) -> bool:
        assert self.active_policy is not None
        return self.active_policy.do_change_rate(rng)

    def get_new_rate(self, interface: HyperdriveReadInterface, rng: Generator) -> FixedPoint:
        assert self.active_policy is not None
        return self.active_policy.get_new_rate(interface, rng)
