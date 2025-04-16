from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Dict
from gymnasium.spaces.utils import flatten_space
from scenario_gym_env import ScenarioGymEnv
import numpy as np

SECONDS_PER_DAY = 24 * 3600

class TimeEncoder(ObservationWrapper):

    def __init__(self, env):
        if not isinstance(env.unwrapped, ScenarioGymEnv):
            raise AttributeError(f'You must pass a ScenarioGymEnv to this wrapper.')
        if len(env.observation_space.shape) > 1:
            raise AttributeError(
                'This wrapper cannot be applied on multi-layer observation spaces.'
            )
        super().__init__(env)
        self.observation_space = flatten_space(
            Dict(old_space=env.observation_space, new_space=self.encoding_space)
        )
        old_observation_desc = env.get_wrapper_attr('observation_desc')
        self.observation_desc = old_observation_desc + self.encoding_desc

    @property
    def encoding_space(self):
        raise NotImplementedError('Implemented by child classes.')

    @property
    def encoding_desc(self):
        raise NotImplementedError('Implemented by child classes')

    def time_in_seconds(self):
        current_itr = self.env.unwrapped.current_itr
        scenario_config = self.env.unwrapped._scenario_config
        reporting_time_step = scenario_config.general_params['reporting_time_step']
        return (current_itr-1) * reporting_time_step

    def encode(self, time_in_seconds):
        raise NotImplementedError('Implemented by child classes')

    def observation(self, obs):
        encoding = self.encode(self.time_in_seconds())
        return np.concatenate([obs, encoding])

class WeekdayEncoder(TimeEncoder):

    @property
    def encoding_space(self):
        return Box(low=0, high=1, shape=(7,), dtype=self.env.unwrapped.float_type)

    @property
    def encoding_desc(self):
        weekdays = [
            'monday', 'tuesday', 'wednesday', 'thursday',
            'friday', 'saturday', 'sunday'
        ]
        return [f'is_{weekday}' for weekday in weekdays]

    def encode(self, time_in_seconds):
        weekday_idx = (time_in_seconds // SECONDS_PER_DAY) % 7
        bin_encoding = np.zeros(
            shape=(7,), dtype=self.env.unwrapped.float_type
        )
        bin_encoding[weekday_idx] = 1
        return bin_encoding

class TimeOfDayEncoder(TimeEncoder):

    @property
    def encoding_space(self):
        return Box(low=-1, high=1, shape=(2,), dtype=self.env.unwrapped.float_type)

    @property
    def encoding_desc(self):
        return ['sin_daytime', 'cos_daytime']

    def encode(self, time_in_seconds):
        part_of_period = time_in_seconds * 2 * np.pi / SECONDS_PER_DAY
        periodic_encoding = np.array(
            [np.sin(part_of_period), np.cos(part_of_period)],
            dtype=self.env.unwrapped.float_type
        )
        return periodic_encoding

class EpisodeStepsEncoder(TimeEncoder):

    @property
    def encoding_space(self):
        return Box(
            low=0,
            high=self.env.unwrapped.max_itr_till_truncation,
            shape=(1,),
            dtype=self.env.unwrapped.float_type
        )

    @property
    def encoding_desc(self):
        return ['episode_steps']

    def encode(self, time_in_seconds):
        res = np.array(
            [self.env.unwrapped.current_itr],
            dtype=self.env.unwrapped.float_type
        )
        return res

