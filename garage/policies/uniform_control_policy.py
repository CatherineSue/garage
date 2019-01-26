"""Uniform Control Policy."""
import numpy as np

from garage.core import Serializable
from garage.misc.overrides import overrides
from garage.policies import Policy


class UniformControlPolicy(Policy, Serializable):
    """Uniform Control Policy."""

    def __init__(
            self,
            env_spec,
    ):
        Serializable.quick_init(self, locals())
        super().__init__(env_spec=env_spec)

    @overrides
    def get_action(self, observation):
        """Return action."""
        return self.action_space.sample(), dict()

    @overrides
    def get_params_internal(self, **tags):
        """Return a list of policy internal params."""
        return []

    @overrides
    def get_param_values(self, **tags):
        """Return values of params."""
        return np.random.uniform(-1, 1, 1000)
