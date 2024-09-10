"""Defining policy classes for controlling variable rates."""

from dataclasses import dataclass

import numpy as np
from agent0.ethpy.hyperdrive import HyperdriveReadInterface
from fixedpointmath import FixedPoint


class VariableRatePolicy:
    @dataclass(kw_only=True)
    class Config:
        rate_change_probability: float = 0.1

    def __init__(self, config: Config | None = None) -> None:
        if config is None:
            config = self.Config()
        self.config = config

    def do_change_rate(self) -> bool:
        """Function defining when to change the rate.

        This function gets called every `step`, and returns `True` if the rate should be changed.
        The default behavior is to change the rate based on `rate_change_probability`.

        Returns
        -------
        bool
            Whether or not to change the rate
        """
        if np.random.rand() < self.config.rate_change_probability:
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
        raise NotImplemented


class RandomNormalVariableRate(VariableRatePolicy):
    @dataclass(kw_only=True)
    class Config(VariableRatePolicy.Config):
        loc: float = 1.0
        scale: float = 0.01

    def get_new_rate(self, interface: HyperdriveReadInterface) -> FixedPoint:
        # Type narrow
        assert isinstance(self.config, RandomNormalVariableRate.Config)
        current_rate = interface.get_variable_rate()
        # narrow type
        assert current_rate is not None
        # new rate is random & between 10x and 0.1x the current rate
        return current_rate * FixedPoint(
            np.minimum(10.0, np.maximum(0.1, np.random.normal(loc=self.config.loc, scale=self.config.scale)))
        )
