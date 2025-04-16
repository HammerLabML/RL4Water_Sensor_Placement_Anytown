from scenario_gym_env import ScenarioGymEnv
from epyt_flow.simulation import ScenarioSimulator, ModelUncertainty
from epyt_flow.simulation.events.actuator_events import PumpStateEvent,\
ActuatorConstants
from epyt_flow.uncertainty import PercentageDeviationUncertainty
from gymnasium.envs.registration import EnvSpec
from gymnasium.spaces import Box, MultiDiscrete, Discrete
from abc import abstractmethod
import numpy as np
import pandas as pd
import yaml
from os import path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from stable_baselines3.common.env_checker import check_env
import itertools
from collections import deque
from water_benchmark_hub import load

import sim_util
import gym_util
from objective_calculator import ObjectiveCalculator

class PumpControlEnv(ScenarioGymEnv):
    '''Abstract class for pump control environments'''

    metadata = {'render_modes': [None, 'rgb_array'], 'render_fps': 30}

    def __init__(self, scenario_configs, initial_pump_speeds, objective_calculator,
            model_uncertainty=None, max_itr_till_truncation=int(1e6),
            render_mode='rgb_array', float_type=np.float32):
        super().__init__(
            scenario_configs,
            model_uncertainty=model_uncertainty,
            max_itr_till_truncation=max_itr_till_truncation,
            render_mode=render_mode,
            float_type=float_type
        )
        if render_mode not in self.metadata['render_modes']:
            raise ValueError(
                f"render_mode was {render_mode}, but must be one of "
                f"{self.metadata['render_modes']}"
            )
        self.pumps = self._scenario_config.sensor_config.pumps
        self.initial_pump_speeds = initial_pump_speeds.copy()
        self._current_pump_speeds = initial_pump_speeds.copy()
        self.objective_calculator = objective_calculator

        self.tank_data = self._scenario_sim.epanet_api.getNodeTankData().to_dict()

    @classmethod
    def from_files(cls, network_file_or_files, initial_pump_speeds_file,
            network_constraints_file, objective_weights_file,
            speed_increment=None, max_pump_speed=None,
            max_speed_change=None, model_uncertainty_file=None,
            standard_sim_options=dict(),
            **kwargs):
        if isinstance(network_file_or_files, str):
            # If only one file is given as a string, produce a single config
            network_file = network_file_or_files
            dummy_sim = sim_util.standard_sim(network_file, **standard_sim_options)
            scenario_configs = dummy_sim.get_scenario_config()
            dummy_sim.close()
        else:
            # Create a round-robin generator producing ScenarioConfigs
            # of each network file in turn
            finite_config_generator = (
                sim_util.get_sc_and_close(
                    sim_util.standard_sim(network_file, **standard_sim_options)
                )
                for network_file in network_file_or_files
            )
            scenario_configs = itertools.cycle(finite_config_generator)
            # The first network file will be used to create the ObjectiveCalculator
            network_file = network_file_or_files[0]
        with open(initial_pump_speeds_file, 'r') as fp:
            initial_pump_speeds = yaml.safe_load(fp)
        initial_pump_speeds = {
            str(k): float(v) for (k, v) in initial_pump_speeds.items()
        }
        objective_calculator = ObjectiveCalculator.from_files(
            network_constraints_file, objective_weights_file, network_file
        )
        if model_uncertainty_file is not None:
            with open(model_uncertainty_file, 'r') as fp:
                model_uncertainty = ModelUncertainty.load_from_json(fp.read())
        else:
            model_uncertainty = None
        kwargs.update(dict(model_uncertainty=model_uncertainty))
        if cls==DiscretePumpControlEnv:
            if speed_increment is None:
                raise ValueError(
                    f'speed_increment was None, but must be given '
                    f'for DiscretePumpControlEnv'
                )
            return cls(
                scenario_configs, initial_pump_speeds, objective_calculator,
                speed_increment, **kwargs
            )
        if cls==ContinuousPumpControlEnv:
            if max_pump_speed==None:
                raise ValueError(
                    f'max_pump_speed was None, but must be given '
                    f'for ContinuousPumpControlEnv'
                )
            return cls(
                scenario_configs, initial_pump_speeds, objective_calculator,
                max_pump_speed, **kwargs
            )
        if cls==RestrictedContinuousPCE:
            if max_pump_speed is None:
                raise ValueError(
                    f'max_pump_speed was None, but must be given '
                    f'for ContinuousPumpControlEnv'
                )
            if max_speed_change is None:
                raise ValueError(
                    f'max_speed_change was None, but must be given '
                    f'for RestrictedContinuousPCE'
                )
            return cls(
                scenario_configs, initial_pump_speeds, objective_calculator,
                max_pump_speed, max_speed_change,
                **kwargs
            )


    @property
    def current_pump_speeds(self):
        return self._current_pump_speeds

    def reset(self, seed=None, options=None, keep_scenario=False):
        '''Set the initial pump speed.'''
        self._finish_sim_generator()
        if not keep_scenario:
            self._next_scenario()
        self._next_sim_generator()
        # Make sure that there are no control rules interfering with the pumps
        self._scenario_sim.epanet_api.deleteControls()
        # Check if the necessary sensors are installed to compute the objective
        sensor_config = self._scenario_config.sensor_config
        self.objective_calculator.check_pump_and_tank_flow_sensors(sensor_config)
        self.objective_calculator.check_pump_efficiency_sensors(sensor_config)
        for pump_id, speed in self.initial_pump_speeds.items():
            self.set_pump_speed(pump_id, speed)
        self._current_pump_speeds = self.initial_pump_speeds.copy()
        return self._next_sim_itr()

    def calculate_reward(self, scada, centered_reward=True):
        """
        Compute the reward and further info using self.objective_calculator

        @param scada: ScadaData object, current sensor readings

        @param centered_reward: bool, default=True
        If True, 0.5 is subtracted from the reward such that it lies in the
        range [-0.5, 0.5] rather than [0, 1]

        @returns
        - reward: float, the computed reward
        - reward_info: dict with the following keys
            - pressure_obj: float in [0, 1], pressure objective
            - tank_obj: float in [0, 1], objective regarding tank in-/outflow
            - pump_efficiency_obj: float in [0, 1], pump efficiecy objective
            - reward: float in [0, 1] (or [-0.5, 0.5] if centered)
                total reward, same as the one returned separately
            - centered: bool, cenetered_reward parameter
        """
        reward_info = self.objective_calculator.full_objective(
            scada, return_all=True, centered=centered_reward
        )
        reward_info['centered'] = centered_reward
        return reward_info['reward'], reward_info

    def _mark_initial_speeds(self, patches):
        '''Draw non-filled rectangles marking initial speeds.'''
        zipped = zip(
            patches,
            self.initial_pump_speeds.values(),
            self.current_pump_speeds.values()
        )
        for patch, initial_speed, current_speed in zipped:
            x0 = patch.xy[0]
            x1 = x0 + patch._width
            color = 'black' if initial_speed >= current_speed else 'yellow'
            plt.plot(
                [x0, x0, x1, x1],
                [0, initial_speed, initial_speed, 0],
                color=color, linewidth=2
            )

    def render_rgb(self):
        fig = plt.figure()
        speeds = list(self._current_pump_speeds.values())
        x_pos = list(range(len(speeds)))
        bar_container = plt.bar(x_pos, speeds, color='green')
        labels = list(self._current_pump_speeds.keys())
        plt.xticks(x_pos, labels)
        self._mark_initial_speeds(bar_container.patches)
        plt.suptitle('Current Pump Speeds')
        canvas = FigureCanvas(fig)
        canvas.draw()
        buffer = canvas.buffer_rgba()
        width, height = fig.get_size_inches() * fig.get_dpi()
        rgba_array = np.frombuffer(buffer, dtype='uint8')
        rgba_array = rgba_array.reshape(int(height), int(width), 4)
        rgb_array = rgba_array[:, :, :3]
        plt.close(fig)
        return rgb_array

    def render(self):
        if self.render_mode not in self.metadata['render_modes']:
            raise ValueError(
                f"render_mode was {self.render_mode}, but must be one of "
                f"{self.metadata['render_modes']}"
            )
        if self.render_mode=='rgb_array':
            return self.render_rgb()
        else: # self.render_mode is None
            pass

    @abstractmethod
    def step(self, *actions):
        '''To be implemented in subclasses'''
        raise NotImplementedError()

class DiscretePumpControlEnv(PumpControlEnv):

    metadata = {'render_modes': [None, 'rgb_array'], 'render_fps': 30}

    def __init__(self, scenario_configs, initial_pump_speeds,
            objective_calculator, speed_increment,
            model_uncertainty=None,
            max_itr_till_truncation=int(1e6),
            render_mode='rgb_array', float_type=np.float32,
            multidiscrete_action_space=True):
        super().__init__(
            scenario_configs, initial_pump_speeds, objective_calculator,
            model_uncertainty=model_uncertainty,
            max_itr_till_truncation=max_itr_till_truncation,
            render_mode=render_mode, float_type=float_type
        )
        n_sensors = self._observation_space.shape[0]
        n_pumps = len(self.pumps)
        # If only speed changes are applied, the current speed is not
        # part of the action, and must therefore be part of the observation
        self._observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(n_pumps + n_sensors,),
            dtype=self.float_type
        )
        # each pump speed can be decreased, increased or left alone
        if multidiscrete_action_space:
            self._action_space = MultiDiscrete([3] * n_pumps)
        else:
            # Used internally
            self._multidiscrete_action_space = MultiDiscrete([3] * n_pumps)
            # Used for interfaces
            self._action_space = Discrete(3**n_pumps)
        self.speed_increment = speed_increment

    @property
    def action_space(self):
        return self._action_space

    @classmethod
    def from_files(cls, network_file_or_files, initial_pump_speeds_file,
            network_constraints_file, objective_weights_file,
            speed_increment, model_uncertainty_file=None,
            multidiscrete_action_space=True, **kwargs):
        return super(DiscretePumpControlEnv, cls).from_files(
            network_file_or_files, initial_pump_speeds_file,
            network_constraints_file, objective_weights_file,
            speed_increment=speed_increment,
            model_uncertainty_file=model_uncertainty_file,
            multidiscrete_action_space=multidiscrete_action_space,
            **kwargs
        )

    def _scada2obs_and_info(self, scada, clip_negative_pressures=True):
        obs, info = super()._scada2obs_and_info(
            scada, clip_negative_pressures=clip_negative_pressures
        )
        pump_speeds_arr = np.array(
            list(self._current_pump_speeds.values()),
            dtype=self.float_type
        )
        obs = np.concat([obs, pump_speeds_arr])
        pump_speed_desc = np.array(
            [['pump_speed', pump, ''] for pump in self.pumps]
        )
        info['data_desc'] = np.vstack([info['data_desc'], pump_speed_desc])
        return obs, info

    def step(self, pump_speed_changes):
        '''
        Change pump speed values and observe results.

        @param pump_speed_changes: If multidiscrete_action_space was True at
        object initialization (the default), this must be an np.array of shape
        len(self.pumps).  The values must be integers from 0 to 2 ordered in
        the same way as self.pumps. Meanings are as follows:
        0: do not change the pump speed
        1: decrease the pump speed
        2: increase the pump speed
        Increments and decrements are done by self.speed_increment
        Note: Results of decrements will be clipped to 1e-10.
        If multidiscrete_action_space was False at initialization, a discrete
        action space is used instead. A value from the discrete action space
        must be passed which is internally converted into a value of the actual
        underlying multidiscrete action space using a ternary encoding (see
        gym_util).

        @returns
        - observation (ScadaData): observations for the next time step
        - reward (float): reward function to train an RL agent
        - terminated (bool): If True, this was the last iteration.
        terminated is only returned of self.autoreset is False.
        '''
        if not self._action_space.contains(pump_speed_changes):
            raise ValueError(
                f'speed_changes {pump_speed_changes} not in self.action_space '
                f'{self._action_space}'
            )
        if isinstance(self._action_space, Discrete):
            pump_speed_changes = gym_util.to_multidiscrete(
                pump_speed_changes, self._multidiscrete_action_space
            )
        new_pump_speeds = self._current_pump_speeds.copy()
        for pump, speed_change in zip(self.pumps, pump_speed_changes, strict=True):
            if speed_change==0: # Don't change the speed
                pass
            elif speed_change==1: # Decrease the speed
                new_pump_speed = np.clip(
                    self._current_pump_speeds[pump] - self.speed_increment,
                    a_min=1e-10, a_max=None
                )
                new_pump_speeds[pump] = new_pump_speed
            elif speed_change==2: # Increase the speed
                new_pump_speeds[pump] += self.speed_increment
            self.set_pump_speed(pump, new_pump_speeds[pump])
        self._current_pump_speeds = new_pump_speeds.copy()
        return self._next_sim_itr()

class ContinuousPumpControlEnv(PumpControlEnv):

    metadata = {'render_modes': [None, 'rgb_array'], 'render_fps': 30}

    def __init__(self, scenario_configs, initial_pump_speeds,
            objective_calculator, max_pump_speed,
            min_pump_speed=0, model_uncertainty=None,
            max_itr_till_truncation=int(1e6),
            render_mode='rgb_array', float_type=np.float32):
        super().__init__(
            scenario_configs, initial_pump_speeds, objective_calculator,
            model_uncertainty=model_uncertainty,
            max_itr_till_truncation=max_itr_till_truncation,
            render_mode=render_mode,
            float_type=float_type
        )
        if min_pump_speed>=max_pump_speed:
            raise ValueError('min_pump_speed must be smaller than max_pump_speed')
        self._max_pump_speed = max_pump_speed
        self._min_pump_speed = min_pump_speed
        self._action_space = Box(
            low=1e-12, high=max_pump_speed,
            shape=(len(self.pumps),)
        )
        # The history is important to compute the speed smoothness reward
        current_pump_speeds_arr = np.array(
            list(self._current_pump_speeds.values()),
            dtype=self.float_type
        )
        self._speed_history = deque(
            [current_pump_speeds_arr for _ in range(3)],
            maxlen=3
        )

    @property
    def max_pump_speed(self):
        return self._max_pump_speed

    @property
    def action_space(self):
        return self._action_space

    @property
    def speed_history(self):
        return self._speed_history

    @classmethod
    def from_files(cls, network_file_or_files, initial_pump_speeds_file,
            network_constraints_file, objective_weights_file,
            max_pump_speed, model_uncertainty_file=None,
            **kwargs):
        return super(ContinuousPumpControlEnv, cls).from_files(
            network_file_or_files, initial_pump_speeds_file,
            network_constraints_file, objective_weights_file,
            max_pump_speed=max_pump_speed,
            model_uncertainty_file=model_uncertainty_file,
            **kwargs
        )

    def calculate_reward(self, scada, centered_reward=True):
        """
        Compute the reward and further info using self.objective_calculator

        @param scada: ScadaData object, current sensor readings

        @param centered_reward: bool, default=True
        If True, 0.5 is subtracted from the reward such that it lies in the
        range [-0.5, 0.5] rather than [0, 1]

        @returns
        - reward: float, the computed reward
        - reward_info: dict with the following keys
            - pressure_obj: float in [0, 1], pressure objective
            - tank_obj: float in [0, 1], objective regarding tank in-/outflow
            - pump_efficiency_obj: float in [0, 1], pump efficiecy objective
            - smoothness_obj: float in [0, 1], objective for smooth speed curves
            - pump_price_obs: float mostly in [0, 1], objective measuring energy
              cost (1 meaning no cost, 0 meaning high cost). The value might be
              negative because the normalization is done empirically (compare
              sim_util.max_pump_price)
            - reward: float in [0, 1] (or [-0.5, 0.5] if centered)
                total reward, same as the one returned separately
            - centered: bool, cenetered_reward parameter
        """
        reward_info = self.objective_calculator.full_objective(
            scada,
            speed_history=self._speed_history,
            max_pump_speed=self.max_pump_speed,
            return_all=True,
            centered=centered_reward
        )
        reward_info['centered'] = centered_reward
        return reward_info['reward'], reward_info

    def step(self, new_pump_speeds):
        '''
        Set the pump speeds and step the environment

        @param new_pump_speeds, np.ndarray, pump speeds to set
        The speeds must lie in self.action_space

        @returns
        - obs: np.ndarray in self.observation_space, current sensor readings
        - reward: float, reward (see self.calculate_reward)
        - terminated: bool, whether the simulation has ended
        - truncated: bool, whether the episode was truncated
        - info: dictionary containing the following keys:
          - data_desc: information about measurement types and locations
          - reward_info: separate reward components
          - sim_time: current time since simulation start in seconds
        '''
        # Avoid numerical problem arising from action scaling
        if (new_pump_speeds>=0).all():
            new_pump_speeds = np.clip(new_pump_speeds, a_min=1e-12, a_max=None)
        if not self._action_space.contains(new_pump_speeds):
            raise ValueError(
                f'new_pump_speeds {new_pump_speeds} not in self.action_space '
                f'{self._action_space}'
            )
        new_pump_speeds = dict(zip(self.pumps, new_pump_speeds, strict=True))
        for pump, new_pump_speed in new_pump_speeds.items():
            self.set_pump_speed(pump, new_pump_speed)
            # TODO: Implement pump status change based on min_pump_speed.
            # Before this, the set_pump_status method in scenario_gym_env
            # must be adapted to be compatible with patterns.
        self._current_pump_speeds = new_pump_speeds.copy()
        current_pump_speeds_arr = np.array(
            list(self._current_pump_speeds.values()),
            dtype=self.float_type
        )
        self._speed_history.append(current_pump_speeds_arr)
        res = self._next_sim_itr()
        return res

class RestrictedContinuousPCE(ContinuousPumpControlEnv):

    """
    Continuous pump control environment with restricted speed change.

    Between two timesteps, the speed of pumps may only be changed by at most
    max_speed_change. The action space is adapted accordingly, by taking a bounded
    increment or decrement as an action.

    IMPORTANT: This class should always be wrapped with SpeedAwareObservation.
    Otherwise, the agent doesn't know to which current pump speed it's speed
    change is applied.
    """
    metadata = {'render_modes': [None, 'rgb_array'], 'render_fps': 30}

    def __init__(self, scenario_configs, initial_pump_speeds,
            objective_calculator, max_pump_speed,
            max_speed_change,
            min_pump_speed=0,
            model_uncertainty=None,
            max_itr_till_truncation=int(1e6),
            render_mode='rgb_array', float_type=np.float32):
        super().__init__(
            scenario_configs, initial_pump_speeds,
            objective_calculator, max_pump_speed,
            min_pump_speed=min_pump_speed,
            model_uncertainty=model_uncertainty,
            max_itr_till_truncation=max_itr_till_truncation,
            render_mode=render_mode,
            float_type=float_type
        )
        self._max_speed_change = max_speed_change
        self._action_space = Box(
            low=-max_speed_change,
            high=max_speed_change,
            shape=(len(self.pumps),)
        )

    @property
    def max_speed_change(self):
        return self._max_speed_change

    @property
    def action_space(self):
        return self._action_space

    @classmethod
    def from_files(cls, network_file_or_files, initial_pump_speeds_file,
            network_constraints_file, objective_weights_file,
            max_pump_speed, max_speed_change,
            model_uncertainty_file=None,
            **kwargs):
        return super(RestrictedContinuousPCE, cls).from_files(
            network_file_or_files, initial_pump_speeds_file,
            network_constraints_file, objective_weights_file,
            max_pump_speed=max_pump_speed,
            max_speed_change=max_speed_change,
            model_uncertainty_file=model_uncertainty_file,
            **kwargs
        )

    def step(self, speed_changes):
        if speed_changes not in self._action_space:
            raise ValueError(
                f'speed_changes {speed_changes} not in action_space '
                f'{self.action_space}'
            )
        new_pump_speeds = self._current_pump_speeds.copy()
        for pump, speed_change in zip(self.pumps, speed_changes, strict=True):
            new_pump_speed = np.clip(
                self._current_pump_speeds[pump] + speed_change,
                a_min=max([self._min_pump_speed, 1e-12]),
                a_max=self._max_pump_speed
            )
            new_pump_speeds[pump] = new_pump_speed
            self.set_pump_speed(pump, new_pump_speed)
        self._current_pump_speeds = new_pump_speeds.copy()
        return self._next_sim_itr()

def create_matching_env(
        network_file_or_files,
        initial_pump_speeds_file,
        network_constraints_file,
        objective_weights_file,
        continuous,
        max_speed_change=None,
        max_pump_speed=None,
        speed_increment=None,
        **kwargs):
    if continuous:
        if max_speed_change is None:
            pump_control_env = ContinuousPumpControlEnv.from_files(
                network_file_or_files,
                initial_pump_speeds_file,
                network_constraints_file,
                objective_weights_file,
                max_pump_speed=max_pump_speed,
                **kwargs
            )
        else:
            pump_control_env = RestrictedContinuousPCE.from_files(
                network_file_or_files,
                initial_pump_speeds_file,
                network_constraints_file,
                objective_weights_file,
                max_pump_speed=max_pump_speed,
                max_speed_change=max_speed_change,
                **kwargs
            )
    else: # continuous was False, so a version with discrete actions is created
        pump_control_env = DiscretePumpControlEnv.from_files(
            network_file_or_files,
            initial_pump_speeds_file,
            network_constraints_file,
            objective_weights_file,
            speed_increment=speed_increment,
            **kwargs
        )
    return pump_control_env

def net1_env(
        continuous=True,
        max_speed_change = None,
        max_pump_speed = 1.0,
        speed_increment = None,
        objective_weights_file=(
            '../Data/Parameters_for_Optimization/objective_weights_1.yaml'
        ),
        **kwargs):
    """
    Construct standard environment for Net1 with discrete or continuous actions.

    Note the directory from which the .inp-file, the constraints file and the pump
    speeds file are loaded is currently hard-coded.

    @param continuous, bool, whether or not to use continuous actions
    If discrete action are used, three actions are possible for each pump:
    incrementing the speed, decrementing the speed or leaving it the same.
    Otherwise, the action determines the pump speed directly.

    @param max_speed_change, float, optional, default=None
    This option is only used if continuous=True
    It determines the maximum change in speed for successive timesteps. If
    this is given, a RestrictedContinuousPCE is created where actions belong
    to the interval [-max_speed_increment, max_speed_increment] for each pump
    and determine how much and in which direction the speed is changed.
    If this is None (default), the speed can be freely chosen at each
    timestep.

    @param **kwargs, further keyword arguments determining the enviornment
    - max_pump_speed: float, default=6
    - model_uncertainty_file: str, default=None, .json-file to reconstruct an
      epyt_flow.model_uncertainty object to be used in each simulation (e.g.
      for adding demand pattern uncertainty)
    - speed_increment: float, default=0.05, only used in the discrete case
    - multidiscrete_action_space: bool, default=False, only used in the
      discrete case
    See the documentation of DiscretePumpControlEnv for details.

    @returns a PumpControlEnv representing Net1
    """
    network_dir = '../Data/Net1'
    network_file = path.join(network_dir, 'net1.inp')
    network_constraints_file = path.join(network_dir, 'net1_constraints.yaml')
    initial_pump_speeds_file = path.join(network_dir, 'initial_pump_speeds.yaml')
    pump_control_env = create_matching_env(
        network_file,
        initial_pump_speeds_file,
        network_constraints_file,
        objective_weights_file,
        continuous,
        max_speed_change=max_speed_change,
        max_pump_speed=max_pump_speed,
        speed_increment=speed_increment,
        **kwargs
    )
    net1_spec = EnvSpec(
        'net1', entry_point=net1_env,
        max_episode_steps=pump_control_env._max_itr_till_truncation,
        disable_env_checker=True
    )
    pump_control_env.spec = net1_spec
    return pump_control_env

def net1_discrete_env(**kwargs):
    if not 'speed_increment' in kwargs.keys():
        kwargs['speed_increment'] = 0.05
    return net1_env(continuous=False, **kwargs)

def net1_continuous_env(**kwargs):
    return net1_env(continuous=True, **kwargs)

def net1_restricted_continuous_env(**kwargs):
    if not 'max_speed_change' in kwargs.keys():
        kwargs['max_speed_change'] = 0.2
    return net1_env(continuous=True, **kwargs)

def leakdb_env(
        scenario_nrs=None,
        start_scenario=None,
        end_scenario=None,
        continuous=True,
        max_speed_change=None,
        max_pump_speed=1.0,
        speed_increment=None,
        objective_weights_file=(
            '../Data/Parameters_for_Optimization/objective_weights_1.yaml'
        ),
        **kwargs):
    net1_dir = '../Data/Net1'
    leakdb_dir = path.join(net1_dir, 'LeakDB_Versions')
    if scenario_nrs is None:
        try:
            scenario_nrs = list(range(start_scenario, end_scenario+1))
        except TypeError:
            raise AttributeError(
                f'You have to either specify scenario_nrs as a list '
                f'or provide a start and end scenario.'
            )
    network_files = [
        path.join(leakdb_dir, f'Scenario-{i}.inp') for i in scenario_nrs
    ]
    network_constraints_file = path.join(net1_dir, 'net1_constraints.yaml')
    initial_pump_speeds_file = path.join(net1_dir, 'initial_pump_speeds.yaml')
    if 'max_itr_till_truncation' not in kwargs.keys():
        kwargs['max_itr_till_truncation'] = 337 # One week of simulations
    pump_control_env = create_matching_env(
        network_files,
        initial_pump_speeds_file,
        network_constraints_file,
        objective_weights_file,
        continuous,
        max_speed_change=max_speed_change,
        max_pump_speed=max_pump_speed,
        speed_increment=speed_increment,
        **kwargs
    )
    leakdb_spec = EnvSpec(
        'leakdb', entry_point=leakdb_env,
        max_episode_steps=pump_control_env._max_itr_till_truncation,
        disable_env_checker=True
    )
    pump_control_env.spec = leakdb_spec
    return pump_control_env

def net3_env(
        continuous=True,
        max_speed_change = None,
        max_pump_speed = 1.0,
        speed_increment = None,
        objective_weights_file=(
            '../Data/Parameters_for_Optimization/objective_weights_1.yaml'
        ),
        **kwargs):
    """
    Construct standard environment for Net1 with discrete or continuous actions.

    Note the directory from which the .inp-file, the constraints file and the pump
    speeds file are loaded is currently hard-coded.

    @param continuous, bool, whether or not to use continuous actions
    If discrete action are used, three actions are possible for each pump:
    incrementing the speed, decrementing the speed or leaving it the same.
    Otherwise, the action determines the pump speed directly.

    @param max_speed_change, float, optional, default=None
    This option is only used if continuous=True
    It determines the maximum change in speed for successive timesteps. If
    this is given, a RestrictedContinuousPCE is created where actions belong
    to the interval [-max_speed_increment, max_speed_increment] for each pump
    and determine how much and in which direction the speed is changed.
    If this is None (default), the speed can be freely chosen at each
    timestep.

    @param **kwargs, further keyword arguments determining the enviornment
    - max_pump_speed: float, default=6
    - model_uncertainty_file: str, default=None, .json-file to reconstruct an
      epyt_flow.model_uncertainty object to be used in each simulation (e.g.
      for adding demand pattern uncertainty)
    - speed_increment: float, default=0.05, only used in the discrete case
    - multidiscrete_action_space: bool, default=False, only used in the
      discrete case
    See the documentation of DiscretePumpControlEnv for details.

    @returns a PumpControlEnv representing Net3
    """
    network_dir = '../Data/Net3'
    network_file = path.join(network_dir, 'net3.inp')
    network_constraints_file = path.join(network_dir, 'net3_constraints.yaml')
    initial_pump_speeds_file = path.join(network_dir, 'initial_pump_speeds.yaml')
    pump_control_env = create_matching_env(
        network_file,
        initial_pump_speeds_file,
        network_constraints_file,
        objective_weights_file,
        continuous,
        max_speed_change=max_speed_change,
        max_pump_speed=max_pump_speed,
        speed_increment=speed_increment,
        **kwargs
    )
    net3_spec = EnvSpec(
        'net3', entry_point=net3_env,
        max_episode_steps=pump_control_env._max_itr_till_truncation,
        disable_env_checker=True
    )
    pump_control_env.spec = net3_spec
    return pump_control_env

def anytown_env(
        continuous,
        max_speed_change = None,
        max_pump_speed = 4.0,
        speed_increment = None,
        objective_weights_file=(
            '../Data/Parameters_for_Optimization/objective_weights_1.yaml'
        ),
        **kwargs):
    """
    Construct standard environment for Anytown with discrete or continuous actions.

    Note the directory from which the .inp-file, the constraints file and the pump
    speeds file are loaded is currently hard-coded.

    @param continuous, bool, whether or not to use continuous actions
    If discrete action are used, three actions are possible for each pump:
    incrementing the speed, decrementing the speed or leaving it the same.
    Otherwise, the action determines the pump speed directly.

    @param max_speed_change, float, optional, default=None
    This option is only used if continuous=True
    It determines the maximum change in speed for successive timesteps. If
    this is given, a RestrictedContinuousPCE is created where actions belong
    to the interval [-max_speed_increment, max_speed_increment] for each pump
    and determine how much and in which direction the speed is changed.
    If this is None (default), the speed can be freely chosen at each
    timestep.

    @param **kwargs, further keyword arguments determining the enviornment
    - max_pump_speed: float, default=6
    - model_uncertainty_file: str, default=None, .json-file to reconstruct an
      epyt_flow.model_uncertainty object to be used in each simulation (e.g.
      for adding demand pattern uncertainty)
    - speed_increment: float, default=0.05, only used in the discrete case
    - multidiscrete_action_space: bool, default=False, only used in the
      discrete case
    See the documentation of DiscretePumpControlEnv for details.

    @returns a PumpControlEnv representing Anytown
    """
    network_dir = '../Data/Anytown'
    network_file = path.join(network_dir, 'anytown.inp')
    network_constraints_file = path.join(network_dir, 'anytown_constraints.yaml')
    initial_pump_speeds_file = path.join(network_dir, 'initial_pump_speeds.yaml')
    pump_control_env = create_matching_env(
        network_file,
        initial_pump_speeds_file,
        network_constraints_file,
        objective_weights_file,
        continuous,
        max_speed_change=max_speed_change,
        max_pump_speed=max_pump_speed,
        speed_increment=speed_increment,
        **kwargs
    )
    anytown_spec = EnvSpec(
        'anytown', entry_point=anytown_env,
        max_episode_steps=pump_control_env._max_itr_till_truncation,
        disable_env_checker=True
    )
    pump_control_env.spec = anytown_spec
    return pump_control_env

def anytown_discrete_env(**kwargs):
    if 'speed_increment' not in kwargs.keys():
        kwargs['speed_increment'] = 0.05
    return anytown_env(continuous=False, **kwargs)

def anytown_continuous_env(**kwargs):
    return anytown_env(continuous=True, **kwargs)

def anytown_restricted_continuous_env(**kwargs):
    if 'max_speed_change' not in kwargs.keys():
        kwargs['max_speed_change'] = 0.2
    return anytown_env(continuous=True, **kwargs)

def ATM_env(
        continuous=True,
        version=3,
        max_speed_change = None,
        max_pump_speed = 1,
        speed_increment = None,
        objective_weights_file=(
            '../Data/Parameters_for_Optimization/objective_weights_1.yaml'
        ),
        **kwargs):
    """
    Construct environment for modified Anytown with discrete or continuous actions.

    Note the directory from which the .inp-file, the constraints file and the pump
    speeds file are loaded is currently hard-coded. The modified version of
    Anytown (Data/Anytown/ATM.inp) will be used with corresponding initial
    speed setting.

    @param continuous, bool, whether or not to use continuous actions
    If discrete action are used, three actions are possible for each pump:
    incrementing the speed, decrementing the speed or leaving it the same.
    Otherwise, the action determines the pump speed directly.

    @param max_speed_change, float, optional, default=None
    This option is only used if continuous=True
    It determines the maximum change in speed for successive timesteps. If
    this is given, a RestrictedContinuousPCE is created where actions belong
    to the interval [-max_speed_increment, max_speed_increment] for each pump
    and determine how much and in which direction the speed is changed.
    If this is None (default), the speed can be freely chosen at each
    timestep.

    @param **kwargs, further keyword arguments determining the enviornment
    - max_pump_speed: float, default=6
    - model_uncertainty_file: str, default=None, .json-file to reconstruct an
      epyt_flow.model_uncertainty object to be used in each simulation (e.g.
      for adding demand pattern uncertainty)
    - speed_increment: float, default=0.05, only used in the discrete case
    - multidiscrete_action_space: bool, default=False, only used in the
      discrete case
    See the documentation of DiscretePumpControlEnv for details.

    @returns a PumpControlEnv representing the modified Anytown (ATM.inp)
    """
    allowed_versions = [1,2,3,4]
    if version not in allowed_versions:
        raise ValueError(
            f'version was {version} but must be one of {allowed_versions}'
        )
    network_dir = '../Data/Anytown'
    network_file = path.join(network_dir, f'ATM-v{version}.inp')
    network_constraints_file = path.join(network_dir, 'ATM_constraints.yaml')
    initial_pump_speeds_file = path.join(
        network_dir, 'initial_pump_speeds_ATM.yaml'
    )
    pump_control_env = create_matching_env(
        network_file,
        initial_pump_speeds_file,
        network_constraints_file,
        objective_weights_file,
        continuous,
        max_pump_speed=max_pump_speed,
        max_speed_change=max_speed_change,
        speed_increment=speed_increment,
        **kwargs
    )
    ATM_spec = EnvSpec(
        'ATM', entry_point=ATM_env,
        max_episode_steps=pump_control_env._max_itr_till_truncation,
        disable_env_checker=True
    )
    pump_control_env.spec = ATM_spec
    return pump_control_env


def KY16_env(
        continuous=True,
        # version=3,
        max_speed_change = None,
        max_pump_speed = 1,
        speed_increment = None,
        objective_weights_file=(
            '../Data/Parameters_for_Optimization/objective_weights_1.yaml'
        ),
        **kwargs):
    """
    Construct environment for KY16 with discrete or continuous actions.

    Note the constraints file and the pump speeds file are loaded is
    currently hard-coded. The .inp file is currently loaded via Water Benchmark
    Hub, which is inconsistent with the rest of the environments.

    @param continuous, bool, whether or not to use continuous actions
    If discrete action are used, three actions are possible for each pump:
    incrementing the speed, decrementing the speed or leaving it the same.
    Otherwise, the action determines the pump speed directly.

    @param max_speed_change, float, optional, default=None
    This option is only used if continuous=True
    It determines the maximum change in speed for successive timesteps. If
    this is given, a RestrictedContinuousPCE is created where actions belong
    to the interval [-max_speed_increment, max_speed_increment] for each pump
    and determine how much and in which direction the speed is changed.
    If this is None (default), the speed can be freely chosen at each
    timestep.

    @param **kwargs, further keyword arguments determining the enviornment
    - max_pump_speed: float, default=6
    - model_uncertainty_file: str, default=None, .json-file to reconstruct an
      epyt_flow.model_uncertainty object to be used in each simulation (e.g.
      for adding demand pattern uncertainty)
    - speed_increment: float, default=0.05, only used in the discrete case
    - multidiscrete_action_space: bool, default=False, only used in the
      discrete case
    See the documentation of DiscretePumpControlEnv for details.

    @returns a PumpControlEnv representing the KY16
    """
    network = load("Network-KY16")
    network_file = network.load()

    network_dir = '../Data/KY16'
    # network_file = path.join(network_dir, f'ATM-v{version}.inp')
    network_constraints_file = path.join(network_dir, 'KY16_constraints.yaml')
    initial_pump_speeds_file = path.join(
        network_dir, 'initial_pump_speeds.yaml'
    )
    pump_control_env = create_matching_env(
        network_file,
        initial_pump_speeds_file,
        network_constraints_file,
        objective_weights_file,
        continuous,
        max_pump_speed=max_pump_speed,
        max_speed_change=max_speed_change,
        speed_increment=speed_increment,
        **kwargs
    )
    pump_control_env._scenario_sim.remove_all_simple_controls()
    demand_model = pump_control_env._scenario_sim.get_demand_model()
    demand_model['type'] = 'PDA'
    demand_model['pressure_required'] = 0.1
    pump_control_env._scenario_sim.set_general_parameters(demand_model=demand_model)
    return pump_control_env



def NJ1_env(
        continuous=True,
        # version=3,
        max_speed_change = None,
        max_pump_speed = 5,
        speed_increment = None,
        objective_weights_file=(
            '../Data/Parameters_for_Optimization/objective_weights_1.yaml'
        ),
        **kwargs):
    """
    Construct environment for KY16 with discrete or continuous actions.

    Note the constraints file and the pump speeds file are loaded is
    currently hard-coded. The .inp file is currently loaded via Water Benchmark
    Hub, which is inconsistent with the rest of the environments.

    @param continuous, bool, whether or not to use continuous actions
    If discrete action are used, three actions are possible for each pump:
    incrementing the speed, decrementing the speed or leaving it the same.
    Otherwise, the action determines the pump speed directly.

    @param max_speed_change, float, optional, default=None
    This option is only used if continuous=True
    It determines the maximum change in speed for successive timesteps. If
    this is given, a RestrictedContinuousPCE is created where actions belong
    to the interval [-max_speed_increment, max_speed_increment] for each pump
    and determine how much and in which direction the speed is changed.
    If this is None (default), the speed can be freely chosen at each
    timestep.

    @param **kwargs, further keyword arguments determining the enviornment
    - max_pump_speed: float, default=6
    - model_uncertainty_file: str, default=None, .json-file to reconstruct an
      epyt_flow.model_uncertainty object to be used in each simulation (e.g.
      for adding demand pattern uncertainty)
    - speed_increment: float, default=0.05, only used in the discrete case
    - multidiscrete_action_space: bool, default=False, only used in the
      discrete case
    See the documentation of DiscretePumpControlEnv for details.

    @returns a PumpControlEnv representing the KY16
    """
    network = load("Network-NJ1")
    network_file = network.load()

    network_dir = '../Data/NJ1'
    # network_file = path.join(network_dir, f'ATM-v{version}.inp')
    network_constraints_file = path.join(network_dir, 'NJ1_constraints.yaml')
    initial_pump_speeds_file = path.join(
        network_dir, 'initial_pump_speeds.yaml'
    )
    pump_control_env = create_matching_env(
        network_file,
        initial_pump_speeds_file,
        network_constraints_file,
        objective_weights_file,
        continuous,
        max_pump_speed=max_pump_speed,
        max_speed_change=max_speed_change,
        speed_increment=speed_increment,
        **kwargs
    )

    # demand_model = pump_control_env._scenario_sim.get_demand_model()
    # demand_model['type'] = 'PDA'
    # demand_model['pressure_required'] = 0.1
    # pump_control_env._scenario_sim.set_general_parameters(demand_model=demand_model)
    return pump_control_env

def Dtown_env(
        continuous=True,
        # version=3,
        max_speed_change = None,
        max_pump_speed = 5,
        speed_increment = None,
        objective_weights_file=(
            '../Data/Parameters_for_Optimization/objective_weights_1.yaml'
        ),
        **kwargs):
    """
    Construct environment for D-Town with discrete or continuous actions.

    Note the constraints file and the pump speeds file are loaded is
    currently hard-coded. The .inp file is currently loaded via Water Benchmark
    Hub, which is inconsistent with the rest of the environments.

    @param continuous, bool, whether or not to use continuous actions
    If discrete action are used, three actions are possible for each pump:
    incrementing the speed, decrementing the speed or leaving it the same.
    Otherwise, the action determines the pump speed directly.

    @param max_speed_change, float, optional, default=None
    This option is only used if continuous=True
    It determines the maximum change in speed for successive timesteps. If
    this is given, a RestrictedContinuousPCE is created where actions belong
    to the interval [-max_speed_increment, max_speed_increment] for each pump
    and determine how much and in which direction the speed is changed.
    If this is None (default), the speed can be freely chosen at each
    timestep.

    @param **kwargs, further keyword arguments determining the enviornment
    - max_pump_speed: float, default=6
    - model_uncertainty_file: str, default=None, .json-file to reconstruct an
      epyt_flow.model_uncertainty object to be used in each simulation (e.g.
      for adding demand pattern uncertainty)
    - speed_increment: float, default=0.05, only used in the discrete case
    - multidiscrete_action_space: bool, default=False, only used in the
      discrete case
    See the documentation of DiscretePumpControlEnv for details.

    @returns a PumpControlEnv representing the D-Town
    """

    network_dir = '../Data/D-Town'
    network_file = path.join(network_dir, f'dtown.inp')
    network_constraints_file = path.join(network_dir, 'dtown_constraints.yaml')
    initial_pump_speeds_file = path.join(
        network_dir, 'initial_pump_speeds.yaml'
    )
    pump_control_env = create_matching_env(
        network_file,
        initial_pump_speeds_file,
        network_constraints_file,
        objective_weights_file,
        continuous,
        max_pump_speed=max_pump_speed,
        max_speed_change=max_speed_change,
        speed_increment=speed_increment,
        **kwargs
    )

    # demand_model = pump_control_env._scenario_sim.get_demand_model()
    # demand_model['type'] = 'PDA'
    # demand_model['pressure_required'] = 0.1
    # pump_control_env._scenario_sim.set_general_parameters(demand_model=demand_model)
    return pump_control_env

if __name__=='__main__':
    model_uncertainty_file = '../Data/Model_Uncertainties/uncertainty_1.json'
    pump_control_env = net1_env(
        continuous=True,
        model_uncertainty_file=model_uncertainty_file
    )
    check_env(pump_control_env)

    info = pump_control_env._scenario_sim.epanet_api.getCurvesInfo()

    obs, info = pump_control_env.reset()
    plt.figure()
    plt.axis('off')
    img = plt.imshow(pump_control_env.render())
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    reward = info['reward_info']['reward']
    terminated = False
    truncated = False
    while not (terminated or truncated):
        time = info['sim_time']
        speeds = pump_control_env.action_space.sample()
        obs, reward, terminated, truncated, info = pump_control_env.step(speeds)
        img.set_data(pump_control_env.render())
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
    pump_control_env.close()

