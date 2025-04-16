import numpy as np
from gymnasium import Env
from stable_baselines3.common.vec_env import VecEnv

class ConstantAgent():

    def __init__(self, env, constant_speeds):
        self._env = env
        self.constant_speeds = constant_speeds

    def predict(self, obs, **kwargs):
        return self.constant_speeds, None

    def get_env(self):
        return self._env

    def set_logger(self, new_logger):
        # only for compatibility
        pass

class ConstantEqualAgent(ConstantAgent):

    def __init__(self, env, constant_speed):
        if isinstance(env, Env):
            base_env = env.unwrapped
        elif isinstance(env, VecEnv):
            base_env = env.get_attr('unwrapped')[0]
        else:
            raise ValueError(
                f'env must either be a gymnasium.Env '
                f'or a stable_baselines3.common.vec_env.VecEnv '
                f', but {type(env)} was given instead'
            )
        shape = env.action_space.shape
        if isinstance(env, VecEnv):
            shape = (env.num_envs,) + shape
        float_type = base_env.float_type
        constant_speeds = constant_speed * np.ones(shape=shape, dtype=float_type)
        super().__init__(env, constant_speeds)

class RecordedAgent():

    def __init__(self, env, recorded_actions):
        self._env = env
        self._recorded_actions = recorded_actions
        self._action_pointer = 0

    @classmethod
    def from_file(cls, env, actions_file):
        recorded_actions = np.load(actions_file)
        return cls(env, recorded_actions)

    def predict(self, obs, **kwargs):
        res = self._recorded_actions[self._action_pointer]
        self._action_pointer += 1
        if self._action_pointer >= len(self._recorded_actions):
            self._action_pointer = 0
        return res, None

    def get_env(self):
        return self._env

    def set_logger(self, new_logger):
        # only for compatibility
        pass

