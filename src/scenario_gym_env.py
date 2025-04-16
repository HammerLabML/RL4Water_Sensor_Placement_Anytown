"""
Module provides a base class for control environments.

This is a slightly modigied version of ScenarioControlEnv in epyt_flow.gym
"""
from abc import abstractmethod, ABC
from typing import Union, SupportsFloat, Optional, Literal, Generator, List
import warnings
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Space
from epyt_flow.simulation import ScadaData, ScenarioConfig, ModelUncertainty
from epyt_flow.simulation.scada import ScadaDataExport

from my_scenario_simulator import MyScenarioSimulator
import sim_util

class ScenarioGymEnv(Env):

    metadata = {'render_modes': [None], 'render_fps': 30}

    """
    Base class for a control environment challenge.

    Parameters
    ----------
    scenario_config : :class:`~epyt_flow.simulation.scenario_config.ScenarioConfig`
                      or a generator yielding `ScenarioConfig` objects
        Scenario configuration. If a generator is given, a new configuration is
        loaded upon reset. This is useful to run experiments with multiple versions
        of the same environment.
    model_uncertainty: :class:`~epyt_flow.simulation.ModelUncertainty`, optional,
                       default=None
        Uncertainty to apply to the simulation. For example, one might add
        uncertainty to the demand patterns.
    max_itr_till_truncation : `int`, optional, default: 10^12
        Maximum number of iterations before an episode is truncated. Note that
        episodes might end earlier due to termination if the simulation reaches
        its end.
    render_mode : optional, `str` or None, default: None
        How to render the environment. None means no rendering. This is the only
        possible option for this class. More options may be added by subclasses.
        cf. https://gymnasium.farama.org/api/env/#gymnasium.Env.render
    float_type : `np.float32` or `np.float64`, default: `np.float32`
        Type of float to use for observations (and actions if applicable)
    """
    def __init__(
            self,
            scenario_configs: Union[ScenarioConfig, Generator[ScenarioConfig, None, None]],
            model_uncertainty: Optional[ModelUncertainty] = None,
            max_itr_till_truncation: int = int(1e6),
            render_mode: Optional[str] = None,
            float_type: Literal[np.float32, np.float64] = np.float32):
        if isinstance(scenario_configs, ScenarioConfig):
            # If only one ScenarioConfig object is given,
            # create a generator that will always return this object
            self.__scenario_config_generator = (
                scenario_configs for _ in iter(int, 1) # i.e. infinitely often
            )
        else:
            self.__scenario_config_generator = scenario_configs
        self._scenario_config = next(self.__scenario_config_generator)
        self._scenario_sim = MyScenarioSimulator(
            scenario_config=self._scenario_config
        )
        self._sim_generator = None
        self._model_uncertainty = model_uncertainty
        if self._model_uncertainty is not None:
            self._scenario_sim.set_model_uncertainty(self._model_uncertainty)
        self._max_itr_till_truncation = max_itr_till_truncation
        self._max_sim_itr = None
        self._current_itr = 0
        allowed_float_types = [np.float32, np.float64]
        if float_type not in allowed_float_types:
            raise ValueError(
                f'float_type was {float_type}, but must be one of '
                f'{allowed_float_types}'
            )
        self.float_type = float_type
        sensor_config = self._scenario_config.sensor_config
        n_sensors = sim_util.n_sensors(sensor_config)
        self._observation_space = Box(
            low=-np.inf, high=np.inf, shape=[n_sensors], dtype=self.float_type
        )
        self._observation_desc = sim_util.obs_columns_from_sensor_config(
            sensor_config
        )
        # action space has to be implemented by child classes
        self._action_space = None
        # possible render modes may be added by subclasses
        self.render_mode = render_mode

    def __exit__(self, *args):
        self.close()

    @property
    def observation_space(self) -> Space:
        """ The observation space of the environment. """
        return self._observation_space

    @property
    def observation_desc(self) -> List[str]:
        """ Descriptors for all observations. """
        return self._observation_desc.copy()

    @property
    def action_space(self) -> Space:
        """
        The action space of the environment.

        This must be constructed in the __init__ method of every subclass.
        """
        raise NotImplementedError('action_space must be implemented by subclasses')

    @property
    def max_itr_till_truncation(self) -> int:
        """
        Maximum number of iterations before an episode is truncated.

        Note that episodes might end earlier due to termination if the
        simulation reaches its end.
        """
        return self._max_itr_till_truncation

    @property
    def max_sim_itr(self) -> Optional[int]:
        """
        Number of iterations before the simulation reaches its end.

        Note that episodes might end earlier due to truncation if the number of
        iterations exceeds self.max_iter_till_truncation.
        Returns None if the reset method has not been called yet.
        """
        return self._max_sim_itr

    @property
    def current_itr(self) -> int:
        """Returns the current number of iterations within the episode."""
        return self._current_itr

    @property
    def model_uncertainty(self) -> Optional[ModelUncertainty]:
        """
        Returns the model uncertainty. This defines if some parts of the
        network model (e.g. demands, pipe lengths) are changed by noise before
        the simulations are run.
        """
        return self._model_uncertainty

    def close(self) -> None:
        """
        Frees all resources.
        """
        try:
            if self._sim_generator is not None:
                self._sim_generator.send(True)
                next(self._sim_generator)
        except StopIteration:
            pass

        if self._scenario_sim is not None:
            self._scenario_sim.close()

    def _set_max_sim_itr(self):
        '''Set the maximum number of iterations within a simulation.'''
        reporting_time_step = (
            self._scenario_config.general_params['reporting_time_step']
        )
        self._max_sim_itr = (
            self._scenario_sim.get_simulation_duration()
            // reporting_time_step
            + 1
        )
        if self._max_sim_itr <= 1:
            raise RuntimeError(
                'At least 2 simulation steps are necessary to see any result'
                ' from the actions taken.'
            )

    def _finish_sim_generator(self):
        """
        Let the generator generate all its content such that the simulation 
        is properly closed.
        """
        if self._sim_generator is not None:
            _ = [scada for scada in self._sim_generator]

    def _next_scenario(self):
        """
        Get the next scenario from self.__scenario_configs.
        Reconstruct the simulator, if necessary.
        """
        new_scenario_config = next(self.__scenario_config_generator)
        # If a single ScenarioConfig object was used to create the environment,
        # the generator will always return the same object so this condition
        # will be false.
        # Otherwise, the new ScenarioConfig must also be used to create a new
        # ScenarioSimulator
        if (self._scenario_config!=new_scenario_config):
            self._scenario_config = new_scenario_config
            self._scenario_sim.close()
            self._scenario_sim = MyScenarioSimulator(
                scenario_config=self._scenario_config
            )

    def _next_sim_generator(self):
        '''Reconstruct the simulation generator and adjust iteration counters.'''
        self._set_max_sim_itr()
        self._current_itr = 0
        self._sim_generator = self._scenario_sim.run_hydraulic_simulation_as_generator(
            support_abort=True
        )

    def reset(self, seed: Optional[int] = None, info: dict = {}, keep_scenario: bool = False) -> tuple[np.ndarray, dict]:
        """
        Reset the environment (i.e. simulation).

        Parameters
        ----------

        seed : `int`, optional, default: None
            not used. Only present for compatibility reasons
        info : `dict`, optional, default: empty dict
            not used. ONly present for compatibility reasons
        keep_scenario: `bool`, default=False
            If multiple scenarios were passed upon creation, this can be set to True
            to keep the current scenario instead of moving on to the next one.

        Returns
        -------
        obs : :class:`~numpy.ndarray`
            Observations from the first step of the simulation
        info : `dict`
            A dictionary containing additional information
            This has the following keys:
                data_desc : :class:`~numpy.ndarray`
                    Array containing sensor locations, sensor types and units of
                    measurement in the same order as obs
                reward_info : `dict`, components of the reward
                sim_time : `int`
                    time in seconds since the start of the current simulation
        """
        self._finish_sim_generator()
        if not keep_scenario:
            self._next_scenario()
        self._next_sim_generator()
        return self._next_sim_itr()

    def _scada2obs_and_info(self,
            scada: ScadaData,
            clip_negative_pressures: bool = True) -> tuple[np.ndarray, dict]:
        '''
        Convert SCADA data to observations and info as used by gymnasium.

        Parameters
        ----------
        scada : :class:`~epyt_flow.simulation.ScadaData`
            scada data object to convert
        clip_negative_pressures : `bool`, optional, default=True
            If True, negative pressure values are clipped to -1

        Returns
        -------
        obs : :class:`numpy.ndarray`
            The observations measured by sensors. `self.float_type` is used as dtype
        info : `dict`
            A dictionary containing additional information
            This has the following keys:
                data_desc : :class:`~numpy.ndarray`
                    Array containing sensor locations, sensor types and units of
                    measurement in the same order as obs
                sim_time : `int`
                    time in seconds since the start of the current simulation
        '''
        obs = scada.get_data().flatten().astype(self.float_type)
        scada_data_export = ScadaDataExport(f_out='/dev/null')
        info = dict()
        if clip_negative_pressures:
            pressure_idxs = [
                i for i in range(len(self._observation_desc))
                if 'pressure' in self._observation_desc[i]
            ]
            obs[pressure_idxs] = np.clip(
                obs[pressure_idxs], a_min=-1, a_max=None
            )
        info['observation_desc'] = self.observation_desc
        info['sim_time'] = scada.sensor_readings_time[0]
        return obs, info

    def _check_term_trunc(self) -> tuple[bool, bool]:
        '''Check if the environment has terminated or truncated (return 2 bools).'''
        terminated = self._current_itr>=self._max_sim_itr
        truncated = self._current_itr>=self.max_itr_till_truncation
        return terminated, truncated

    @abstractmethod
    def calculate_reward(self, scada: ScadaData) -> tuple[SupportsFloat, dict]:
        """
        Computes the reward based on the current sensor readings.

        Parameters
        -----------

        scada : :class:`~epyt_flow.simulation.ScadaData`
            Observations from sensors

        Returns
        ----------

        reward : `float`
            Reward for the agent
        reward_info : `dict`
            Additional information about the reward
        """
        raise NotImplementedError()

    def _next_sim_itr(self) -> Union[tuple[np.ndarray, dict], tuple[np.ndarray, SupportsFloat, bool, bool, dict]]:
        '''
        Perform the next iteration in the current simulation

        When this is the first iteration, only obs and info are returned

        Returns
        --------
        obs : :class:`numpy.ndarray`
            The observations measured by sensors. `self.float_type` is used as dtype
        reward : `float`
            reward as computed by `self.calculate_reward`
        terminated : `bool`
            whether the episode has terminated
        truncated : `bool`
            whether the episode was truncated because self._max_itr_till_truncation
            was exceeded
        info : `dict`
            A dictionary containing additional information
            This has the following keys:
                data_desc : :class:`~numpy.ndarray`
                    Array containing sensor locations, sensor types and units of
                    measurement in the same order as obs
                sim_time : `int`
                    time in seconds since the start of the current simulation
        '''
        next(self._sim_generator)
        scada = self._sim_generator.send(False)
        reward, reward_info = self.calculate_reward(scada)
        self._current_itr += 1
        obs, info = self._scada2obs_and_info(scada)
        if reward_info:
            info['reward_info'] = reward_info
        if self._current_itr==1:
            return obs, info
        terminated, truncated = self._check_term_trunc()
        return obs, reward, terminated, truncated, info

    def set_pump_status(self, pump_id: str, status: int) -> None:
        """
        Sets the status of a pump.

        Parameters
        ----------
        pump_id : `str`
            ID of the pump for which the status is set.
        status : `int`
            New status of the pump -- either active (i.e. open) or inactive (i.e. closed).

            Must be one of the following constants defined in
            :class:`~epyt_flow.simulation.events.actuator_events.ActuatorConstants`:

                - EN_CLOSED  = 0
                - EN_OPEN    = 1
        """
        if self._scenario_sim.f_msx_in is not None:
            raise RuntimeError("Can not execute actions affecting the hydraulics "+
                               "when running EPANET-MSX")

        pump_idx = self._scenario_sim.epanet_api.getLinkPumpNameID().index(pump_id) + 1
        pump_link_idx = self._scenario_sim.epanet_api.getLinkPumpIndex(pump_idx)

        pattern_idx = self._scenario_sim.epanet_api.getLinkPumpPatternIndex(pump_idx)
        if pattern_idx != 0:
            warnings.warn(f"Can not set pump state of pump {pump_id} because a pump pattern exists")
        else:
            self._scenario_sim.epanet_api.setLinkStatus(pump_link_idx, status)

    def set_pump_speed(self, pump_id: str, speed: float) -> None:
        """
        Sets the speed of a pump.

        Parameters
        ----------
        pump_id : `str`
            ID of the pump for which the pump speed is set.
        speed : `float`
            New pump speed.
        """
        pump_idx = self._scenario_sim.epanet_api.getLinkPumpNameID().index(pump_id)
        pattern_idx = self._scenario_sim.epanet_api.getLinkPumpPatternIndex(pump_idx + 1)

        if pattern_idx == 0:
            warnings.warn(f"No pattern for pump '{pump_id}' found -- a new pattern is created")
            pattern_idx = self._scenario_sim.epanet_api.addPattern(f"pump_speed_{pump_id}")
            self._scenario_sim.epanet_api.setLinkPumpPatternIndex(pump_idx + 1, pattern_idx)

        self._scenario_sim.epanet_api.setPattern(pattern_idx, np.array([speed]))

    def set_valve_status(self, valve_id: str, status: int) -> None:
        """
        Sets the status of a valve.

        Parameters
        ----------
        valve_id : `str`
            ID of the valve for which the status is set.
        status : `int`
            New status of the valve -- either open or closed.

            Must be one of the following constants defined in
            :class:`~epyt_flow.simulation.events.actuator_events.ActuatorConstants`:

                - EN_CLOSED  = 0
                - EN_OPEN    = 1
        """
        valve_idx = self._scenario_sim.epanet_api.getLinkValveNameID().index(valve_id)
        valve_link_idx = self._scenario_sim.epanet_api.getLinkValveIndex()[valve_idx]
        self._scenario_sim.epanet_api.setLinkStatus(valve_link_idx, status)

    def set_node_quality_source_value(self, node_id: str, pattern_id: str,
                                      qual_value: float) -> None:
        """
        Sets the quality source at a particular node to a specific value -- e.g.
        setting the chlorine concentration injection to a specified value.

        Parameters
        ----------
        node_id : `str`
            ID of the node.
        pattern_id : `str`
            ID of the quality pattern at the specific node.
        qual_value : `float`
            New quality source value.
        """
        node_idx = self._scenario_sim.epanet_api.getNodeIndex(node_id)
        pattern_idx = self._scenario_sim.epanet_api.getPatternIndex(pattern_id)
        self._scenario_sim.epanet_api.setNodeSourceQuality(node_idx, 1)
        self._scenario_sim.epanet_api.setPattern(pattern_idx, np.array([qual_value]))

    def render(self) -> Optional[Union[np.ndarray, str]]:
        """
        Renders the environment based on self.render_mode

        The only possible render_mode for this class is None, which results in no
        rendering. Other options for render_mode may be implemented in subclasses.
        cf. https://gymnasium.farama.org/api/env/#gymnasium.Env.render
        """
        if self.render_mode is None:
            pass
        else:
            raise NotImplementedError()

    @abstractmethod
    def step(self, *actions) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        """
        Performs the next step by applying an action and observing
        the consequences (obs, reward, terminated, truncated, info)

        Returns
        -------
        obs : :class:`numpy.ndarray`
            Observations (i.e. sensor readings)
        reward : `float`
            Reward for the agent
        terminated : `bool`
            Whether the simulation has come to its end
        truncated : `bool`
            Whether the number of iterations reached self.max_itr_till_truncation
        info : `dict`
            A dictionary containing additional information
            This has the following keys:
                data_desc : :class:`~numpy.ndarray`
                    Array containing sensor locations, sensor types and units of
                    measurement in the same order as obs
                reward_info : `dict`, components of the reward
                sim_time : `int`
                    time in seconds since the start of the current simulation
        """
        raise NotImplementedError()

