import epyt_flow.utils
from gymnasium import ObservationWrapper
from gymnasium.wrappers import TimeAwareObservation
from epyt_flow.simulation import ScenarioSimulator
import pandas as pd
import numpy as np

import sim_util
import gym_util
from objective_calculator import NetworkConstraints
from pump_control_envs import ContinuousPumpControlEnv

class NormalizeHydraulics(ObservationWrapper):
    """
    Normalize observations based on the type of sensor they come from

    @param: env, gymnasium.Env or stable_baselines3.vec_env.VecEnv
    env must be a ContinuousPumpControlEnv or comprised of those.
    env.objective_calculator.network_constraints should contain max_abs_flows
    and max_pump_prices to enable normalization of flow and energy consumption
    """

    def __init__(self, env):
        if not isinstance(env.unwrapped, ContinuousPumpControlEnv):
            raise AttributeError(
                'You must pass a ContinuousPumpControlEnv to this wrapper'
            )
        if len(env.observation_space.shape) > 1:
            raise ValueError(
                f'For this wrapper, observations must be vectors, '
                f'not matrices or higher order tensors.'
            )
        super().__init__(env)
        objective_calculator = self.env.unwrapped.objective_calculator
        self.network_constraints = objective_calculator.network_constraints
        if self.network_constraints.max_abs_flows is None:
            raise ValueError(
                f'Cannot normalize hydraulics if no maximum absolute flow values '
                f'are given in env.objective_calculator.network_constraints'
            )
        max_abs_flows = self.network_constraints.max_abs_flows.copy()
        max_abs_flows.rename(lambda idx: 'flow_' + idx, inplace=True)
        self.max_abs_flows = max_abs_flows.astype(env.unwrapped.float_type)
        nonzero = lambda series: series[series>0] 
        zero = lambda series: series[series==0]
        tank_data = env.unwrapped.tank_data
        self.max_tank_levels = tank_data['Maximum_Water_Level'].astype(
            env.unwrapped.float_type
        )
        self.min_tank_levels = tank_data['Minimum_Water_Level'].astype(
            env.unwrapped.float_type
        )
        self.tank_diameters = tank_data['Diameter']

        self.nonzero_max_abs_flows = nonzero(self.max_abs_flows)
        if len(zero(self.max_abs_flows)) > 0:
            print(
                f'The maximum absolute flow for the following links was zero '
                f'under empirical evaluation: '
                f'{zero(self.max_abs_flows)}\n'
                f'No normalization will be performed for these links.'
            )
        try:
            self.max_pump_prices = self.network_constraints.max_pump_prices.copy()
            self.max_pump_prices = self.max_pump_prices.astype(
                self.env.unwrapped.float_type
            )
            self.max_pump_prices.rename(
                lambda idx: 'pump_energyconsumption_' + idx,
                inplace=True
            )
            self.nonzero_max_pump_prices = nonzero(self.max_pump_prices)
            if len(zero(self.max_pump_prices)) > 0:
                print(
                    f'WARNING: Maximum pump prices for the following pumps were '
                    f'set to zero: {zero(self.max_pump_prices).index}\n'
                    f'No normalization will be performed on these.'
                )
        # This exception will most likely be caused due to max_pump_prices being None
        # in the objective calculator, which can happen if price optimization
        # is not part of the objective
        except AttributeError as err:
            if self.network_constraints.max_pump_prices is None:
                print(
                    f'WARNING: Cannot normalize pump prices: '
                    f'no maximum given'
                )
                self.max_pump_prices = None
            else:
                raise err
            

    def observation(self, obs):
        """
        Three quantities of the observation are normalized as follows:
        - pressure readings are linearly rescaled such that the lower pressure
          bound is mapped to zero and the upper pressure bound is mapped to
          one.
        - flow measurements are rescaled such that the largest absolute flow is
          mapped to one in positive direction or -1 in negative direction. Note
          that the maximum absolut flow is typically determined empirically, so
          it's not guaranteed to be correct.
        - energy consumption measurements are divided by the maximum possible
          energy consuption. Again, this maximum is typically determined
          empirically, so there is no guarantee for its correctness.

        @param obs, numpy.ndarray, original observations produced by the environment
        """
        observation_desc = self.get_wrapper_attr('observation_desc')
        obs = pd.Series(dict(zip(observation_desc, obs, strict=True)))
        flow_idxs = self.nonzero_max_abs_flows.index
        pressure_idxs = [idx for idx in observation_desc if 'pressure' in idx]
        if self.max_pump_prices is not None:
            price_idxs = self.nonzero_max_pump_prices.index
            if not price_idxs.empty:
                obs[price_idxs] /= self.nonzero_max_pump_prices
        if not flow_idxs.empty:
            obs[flow_idxs] /= self.nonzero_max_abs_flows
        if pressure_idxs:
            obs[pressure_idxs] -= self.network_constraints.min_pressure
            obs[pressure_idxs] /= (
                self.network_constraints.max_pressure
                - self.network_constraints.min_pressure
            )
        tank_idxs = [idx for idx in observation_desc if 'tank' in idx]
        if tank_idxs:
            obs[tank_idxs] = [
                epyt_flow.utils.volume_to_level(float(obs[idx]), diameter)
                for diameter,idx in zip(self.tank_diameters, tank_idxs)
            ]
            obs[tank_idxs] -= self.min_tank_levels
            obs[tank_idxs] /= (self.max_tank_levels - self.min_tank_levels)
        obs = obs.to_numpy()
        if np.isnan(obs).any():
            raise ValueError('obs contains NaN after normlization of hydraulics.')
        return obs

    def denormalize(self, normalized_obs, observation_desc=None):
        """
        Retrieve the original observation from an observation that was normalized with this wrapper.

        @param normalized_obs, numpy.ndarray, the normalized observation
        @param observation_desc, dict, default=None
        If necessary, a different observation description can be provided here.
        This can come in handy if other wrappers have been applied in addition
        to this one which changed the observation description.
        """
        if observation_desc is None:
            observation_desc = self.get_wrapper_attr('observation_desc')
        len_observation_desc = len(observation_desc)
        len_obs = normalized_obs.shape[0]
        if len_observation_desc!=len_obs:
            raise AttributeError(
                f'Missmatch between observation_desc (length {len_observation_desc})'
                f'and observation (length {len_obs})'
            )
        normalized_obs = pd.Series(
            dict(zip(observation_desc, normalized_obs, strict=True))
        )
        flow_idxs = self.nonzero_max_abs_flows.index
        flow_idxs = [idx for idx in flow_idxs if idx in observation_desc]
        pressure_idxs = [idx for idx in observation_desc if 'pressure' in idx]
        if flow_idxs:
            normalized_obs[flow_idxs] *= self.max_abs_flows
        if pressure_idxs:
            normalized_obs[pressure_idxs] *= (
                self.network_constraints.max_pressure
                - self.network_constraints.min_pressure
            )
            normalized_obs[pressure_idxs] += self.network_constraints.min_pressure
        if self.max_pump_prices is not None:
            price_idxs = self.nonzero_max_pump_prices.index
            price_idxs = [idx for idx in price_idxs if idx in observation_desc]
            if price_idxs:
                normalized_obs[price_idxs] *= self.nonzero_max_pump_prices
        tank_idxs = [idx for idx in observation_desc if 'tank' in idx]
        if tank_idxs:
            normalized_obs[tank_idxs] *= (self.max_tank_levels - self.min_tank_levels)
            normalized_obs[tank_idxs] += self.min_tank_levels
            level_to_volume_factors = pd.Series(
                np.pi * (self.tank_diameters/2)**2,
                dtype=self.env.unwrapped.float_type
            )
            normalized_obs[tank_idxs] *= level_to_volume_factors
        obs = normalized_obs.to_numpy()
        return obs

