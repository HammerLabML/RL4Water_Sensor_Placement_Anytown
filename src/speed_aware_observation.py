from gymnasium import ObservationWrapper
from gymnasium.spaces.utils import flatten_space, is_space_dtype_shape_equiv
from gymnasium.spaces import Dict, Box
from pump_control_envs import ContinuousPumpControlEnv
import numpy as np

class SpeedAwareObservation(ObservationWrapper):

    def __init__(self, env, n_timesteps):
        if not isinstance(env.unwrapped, ContinuousPumpControlEnv):
            raise AttributeError(
                'You must pass a ContinuousPumpControlEnv to this wrapper'
            )
        super().__init__(env)
        if n_timesteps > 3:
            raise ValueError(
                f'Only 3 timesteps are recorded in history, '
                f'so at most 3 can be appended to the observation.'
            )
        self.n_timesteps = n_timesteps
        old_space = self.env.observation_space
        flat_version = flatten_space(old_space)
        if not is_space_dtype_shape_equiv(old_space, flat_version):
            raise AttributeError(
                f'Your environment must have a flat observation space before it is '
                f'wrapped with this wrapper. In particular, FrameStackObservation '
                f'must be applied afterwards'
            )
        added_space = Box(
            low=0, high=env.unwrapped.max_pump_speed,
            shape=(n_timesteps*len(env.unwrapped.pumps),)
        )
        new_space = flatten_space((Dict(old=old_space, added=added_space)))
        self.observation_space = new_space
        self.observation_desc = env.get_wrapper_attr('observation_desc') + self.additional_info()


    def additional_info(self):
        pumps = self.env.unwrapped.pumps
        speed_info = []
        for i in reversed(range(0, self.n_timesteps)):
            for pump in pumps:
                speed_info.extend([f'pump_speed_t-{i}' + '_' + pump])

        return speed_info

    def observation(self, obs):
        pump_speeds = np.array(
            self.env.unwrapped.speed_history,
            dtype=self.env.unwrapped.float_type
        )
        pump_speeds = pump_speeds[-self.n_timesteps:]
        return np.concatenate([obs, pump_speeds.flatten()])

