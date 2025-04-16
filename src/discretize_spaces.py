import numpy as np
from gymnasium import ActionWrapper
from gymnasium.spaces import Box, MultiDiscrete, Discrete

from pump_control_envs import ContinuousPumpControlEnv
from gym_util import to_multidiscrete


class MultiDiscreteActionsWrapper(ActionWrapper):

    def __init__(self, env, n_bins: int=15) -> None:
        if not isinstance(env.unwrapped, ContinuousPumpControlEnv):
            raise AttributeError(
                'You must pass a ContinuousPumpControlEnv to this wrapper'
            )
        if n_bins < 2:
            raise ValueError(
                f'Amount of bins must be greater than 1, not {n_bins} '
            )
        super().__init__(env)
        assert isinstance(env.action_space, Box)
        self.high = env.action_space.high
        self.low = env.action_space.low
        self.step_size = (self.high - self.low) / n_bins
        self.dtype = env.action_space.dtype
        self.action_space = MultiDiscrete([n_bins]*np.ones(env.action_space.shape))

    def action(self, action):
        act = action * self.step_size + self.low
        return act.astype(self.dtype)



class DiscreteActionsWrapper(ActionWrapper):

    def __init__(self, env) -> None:
        if not isinstance(env.unwrapped, ContinuousPumpControlEnv):
            raise AttributeError(
                'You must pass a ContinuousPumpControlEnv to this wrapper'
            )

        super().__init__(env)
        assert isinstance(env.action_space, MultiDiscrete)
        self.multi_discrete = env.action_space
        self.action_space = Discrete(np.prod(self.multi_discrete.nvec))

    def action(self, action):
        act = to_multidiscrete(action, self.multi_discrete)
        return act
