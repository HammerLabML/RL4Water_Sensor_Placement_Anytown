from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Dict
from gymnasium.spaces.utils import flatten_space
from scenario_gym_env import ScenarioGymEnv
import numpy as np

SECONDS_PER_DAY = 24 * 3600

class TimeEncoder(ObservationWrapper):
    """
    Abstract class for observation wrappers that encode time

    @param env: ScenarioGymEnv or wrapped version thereof. This includes
    PumpControlEnv and all envs deriving from it. However, the wrapper is not
    implemented for vectorized environments
    """

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
        """
        The observation space (gymnasium.space) of the time encoding.
        This will be merged with the observation space of self.env.
        The method is implemented by child classes.
        """
        raise NotImplementedError('Implemented by child classes.')

    @property
    def encoding_desc(self):
        """
        The observation description (list) of the time encoding.
        Example: ["weekday", "hour", "minute"]
        This is meant to be used as an index, when turning a vector (np.ndarray)
        of encoded time into a pandas Series.
        The method is implemented by child classes.
        """
        raise NotImplementedError('Implemented by child classes')

    def time_in_seconds(self):
        """Return the current simulation time in seconds"""
        current_itr = self.env.unwrapped.current_itr
        scenario_config = self.env.unwrapped._scenario_config
        reporting_time_step = scenario_config.general_params['reporting_time_step']
        return (current_itr-1) * reporting_time_step

    def encode(self, time_in_seconds):
        """Encode time (implemented by child classes)."""
        raise NotImplementedError('Implemented by child classes')

    def observation(self, obs):
        """
        Append a time encoding (self.encode) to the current observation and return
        the concatenated result
        """
        encoding = self.encode(self.time_in_seconds())
        return np.concatenate([obs, encoding])

class WeekdayEncoder(TimeEncoder):
    """
    One-hot encoding of the current day of the week

    @param env: ScenarioGymEnv or wrapped version thereof. This includes
    PumpControlEnv and all envs deriving from it. However, the wrapper is not
    implemented for vectorized environments
    """

    @property
    def encoding_space(self):
        """7-element one-hot vector"""
        return Box(low=0, high=1, shape=(7,), dtype=self.env.unwrapped.float_type)

    @property
    def encoding_desc(self):
        """ ['is_monday', 'is_tuesday', ...] """
        weekdays = [
            'monday', 'tuesday', 'wednesday', 'thursday',
            'friday', 'saturday', 'sunday'
        ]
        return [f'is_{weekday}' for weekday in weekdays]

    def encode(self, time_in_seconds):
        """
        Compute the one-hot encoding of the current day of the week
        i.e. return a 7-element vector containing 6 zeros and a single
        one at index i if it's the i-th weekday.
        """
        weekday_idx = (time_in_seconds // SECONDS_PER_DAY) % 7
        bin_encoding = np.zeros(
            shape=(7,), dtype=self.env.unwrapped.float_type
        )
        bin_encoding[weekday_idx] = 1
        return bin_encoding

class TimeOfDayEncoder(TimeEncoder):
    """
    Angular encoding for the current time of the day.

    @param env: ScenarioGymEnv or wrapped version thereof. This includes
    PumpControlEnv and all envs deriving from it. However, the wrapper is not
    implemented for vectorized environments
    """

    @property
    def encoding_space(self):
        """2 elements raning from -1 to 1"""
        return Box(low=-1, high=1, shape=(2,), dtype=self.env.unwrapped.float_type)

    @property
    def encoding_desc(self):
        """sine and cosine of the current daytime"""
        return ['sin_daytime', 'cos_daytime']

    def encode(self, time_in_seconds):
        """
        Compute an angular encoding of the current time of day.
        Take a sine- and cosine function with a 24 hour frequency
        and determine the function values of both for the current time.
        This encoding has the advantage that 0:00 and 24:00 naturally fall
        together, which is also reflected in the metric (i.e. the encoding
        of 23:59 is very close to the one of 00:00).
        """
        part_of_period = time_in_seconds * 2 * np.pi / SECONDS_PER_DAY
        periodic_encoding = np.array(
            [np.sin(part_of_period), np.cos(part_of_period)],
            dtype=self.env.unwrapped.float_type
        )
        return periodic_encoding

class EpisodeStepsEncoder(TimeEncoder):
    """
    Use the number of steps into the episode as a time encoding.

    @param env: ScenarioGymEnv or wrapped version thereof. This includes
    PumpControlEnv and all envs deriving from it. However, the wrapper is not
    implemented for vectorized environments
    """

    @property
    def encoding_space(self):
        """A single element between 0 and max_number_of_steps."""
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
        """Return a 1-element array containing the current step."""
        res = np.array(
            [self.env.unwrapped.current_itr],
            dtype=self.env.unwrapped.float_type
        )
        return res

