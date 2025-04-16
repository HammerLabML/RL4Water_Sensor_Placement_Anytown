import numpy as np
import pandas as pd
from scipy.odr import exponential

import sim_util
from yaml_serializable import YAMLSerializable
import yaml

from my_scenario_simulator import MyScenarioSimulator

W_HYDRAULICALLY_UNSTABLE = 1

class NetworkConstraints(YAMLSerializable):

    def __init__(self, min_pressure, max_pressure, max_pump_efficiencies,
            max_pump_prices=None, max_abs_flows=None):
        self.min_pressure = min_pressure
        self.max_pressure = max_pressure
        self.max_pump_efficiencies = self._to_series(max_pump_efficiencies)
        self.max_pump_prices = self._to_series(max_pump_prices)
        self.max_abs_flows = self._to_series(max_abs_flows)

    def _to_series(self, input_dict):
        if input_dict is None:
            return None
        res = pd.Series(input_dict)
        res.index = res.index.astype(str)
        return res

    def to_dict(self):
        max_pump_efficiencies = {
            p: float(e) for (p, e) in self.max_pump_efficiencies.items()
        }
        if self.max_pump_prices is not None:
            max_pump_prices = {
                p: float(e) for (p, e) in self.max_pump_prices.items()
            }
        else:
            max_pump_prices = None
        if self.max_abs_flows is not None:
            max_abs_flows = {
                l: float(f) for (l,f) in self.max_abs_flows.items()
            }
        else:
            max_abs_flows = None
        res = {
            "min_pressure": float(self.min_pressure),
            "max_pressure": float(self.max_pressure),
            "max_pump_efficiencies": max_pump_efficiencies,
            "max_abs_flows": max_abs_flows,
            "max_pump_prices": max_pump_prices
        }
        return res

class Penalties(YAMLSerializable):

    def __init__(self, pressure_violation=0, speed_change=0, hydraulic_instability=0):
        self._check_validity(
            [pressure_violation, speed_change, hydraulic_instability]
        )
        self.pressure_violation = pressure_violation
        self.speed_change = speed_change
        self.hydraulic_instability = hydraulic_instability

    def _check_validity(self, penalty_values):
        for value in penalty_values:
            if value < 0:
                raise ValueError('Penalties must be non-negative')

    @classmethod
    def all_zero(cls):
        return cls()

    def to_dict(self):
        res = {
            'pressure_violation': float(self.pressure_violation),
            'speed_change': float(self.speed_change),
            'hydraulic_instability': float(self.hydraulic_instability)
        }
        return res

class Weights(YAMLSerializable):

    def __init__(self, pressure_violation=0, abs_tank_flow=0, pump_efficiency=0,
            speed_smoothness=0, pump_price=0):
        self._check_validity([
            pressure_violation, abs_tank_flow, pump_efficiency, speed_smoothness,
            pump_price
        ])
        self.pressure_violation = pressure_violation
        self.abs_tank_flow = abs_tank_flow
        self.pump_efficiency = pump_efficiency
        self.speed_smoothness = speed_smoothness
        self.pump_price = pump_price

    def _check_validity(self, weights):
        if not np.array(weights).sum()==1:
            raise ValueError("Weights don't sum to one.")

    def to_dict(self):
        res = {
            'pressure_violation': float(self.pressure_violation),
            'abs_tank_flow': float(self.abs_tank_flow),
            'pump_efficiency': float(self.pump_efficiency),
            'speed_smoothness': float(self.speed_smoothness),
            'pump_price': float(self.pump_price)
        }
        return res

class Modifiers(YAMLSerializable):

    def __init__(self, pressure_mod='step', tank_mod=None,
            pump_efficiency_mod=None, smoothness_mod='curvature',
            pump_price_mod=None):
        self.check_pressure_mod(pressure_mod)
        self.pressure_mod = pressure_mod
        self.tank_mod = tank_mod
        self.pump_efficiency_mod = pump_efficiency_mod
        self.check_smoothness_mod(smoothness_mod)
        self.smoothness_mod = smoothness_mod
        self.pump_price_mod = pump_price_mod

    def check_pressure_mod(self, pressure_mod):
        allowed_pressure_mods = [
            'step', 'linear', 'exponential', 'linear+step', 'exponential+step'
        ]
        if pressure_mod not in allowed_pressure_mods:
            raise ValueError(
                f'pressure_mod was {pressure_mod}, '
                f'but must be one of {allowed_pressure_mods}'
            )

    def check_smoothness_mod(self, smoothness_mod):
        allowed_smoothness_mods = ['curvature', 'zigzag']
        if smoothness_mod not in allowed_smoothness_mods:
            raise ValueError(
                f'smoothness_mod was {smoothness_mod}, '
                f'but must be one of {allowed_smoothness_mods}'
            )

    @classmethod
    def default(cls):
        return cls()

    def to_dict(self):
        res = {
            'pressure_mod': self.pressure_mod,
            'tank_mod': self.tank_mod,
            'pump_efficiency_mod': self.pump_efficiency_mod,
            'smoothness_mod': self.smoothness_mod,
            'pump_price_mod': self.pump_price_mod
        }
        return res

class ObjectiveParameters(YAMLSerializable):

    def __init__(self,
            weights, penalties=Penalties.all_zero(), modifiers=Modifiers.default()):
        self.weights = weights
        self.penalties = penalties
        self.modifiers = modifiers

    @classmethod
    def from_dict(cls, input_dict):
        '''
        Construct weights, penalties and modifiers from the corresponding
        sub-dictionaries and plug them in the constructor.
        '''
        matching_class = {
            'weights': Weights,
            'penalties': Penalties,
            'modifiers': Modifiers
        }
        return cls(
            **{k: matching_class[k].from_dict(v) for (k, v) in input_dict.items()}
        )

    def to_dict(self):
        res = {
            'weights': self.weights.to_dict(),
            'penalties': self.penalties.to_dict(),
            'modifiers': self.modifiers.to_dict()
        }
        return res

class ObjectiveCalculator():

    def __init__(self, network_constraints, objective_parameters,
            pumps, tank_connections):
        self.network_constraints = network_constraints
        self.params = objective_parameters
        self.pumps = pumps
        self.tank_connections = tank_connections

    @classmethod
    def from_files(cls, network_constraints_file,
            objective_parameters_file, network_file,
            **kwargs):
        network_constraints = NetworkConstraints.from_yaml(network_constraints_file)
        objective_parameters = ObjectiveParameters.from_yaml(
            objective_parameters_file
        )
        dummy_sim = MyScenarioSimulator(f_inp_in=network_file)
        topology = dummy_sim.get_topology()
        pumps = topology.get_all_pumps()
        tank_connections = sim_util.get_tank_connections(topology)
        dummy_sim.close()
        res = cls(
            network_constraints, objective_parameters,
            pumps, tank_connections,
            **kwargs
        )
        return res

    def to_dict(self):
        res = {
            "network_constraints": self.network_constraints.to_dict(),
            "objective_parameters": self.params.to_dict(),
            "pumps": self.pumps,
            "tank_connections": self.tank_connections
        }
        return res

    def to_yaml(self, path):
        with open(path, 'w') as fp:
            yaml.dump(self.to_dict(), fp)

    def __repr__(self):
        return yaml.dump(self.to_dict())

    def pressure_objective(self, pressures, mod=None):
        '''
        Measure satisfaction of pressure bounds.

        @param pressures, np.ndarray,
        row indices (axis=0) correspond to timesteps while column indices (axis=1)
        correspond to nodes.

        @param mod, str, defaults to self.params.modifiers.pressure_mod
        describes the type of objective function. Possible options are:
        - 'step': Give a nodal reward of 1 for pressure satisfaction at all timesteps
          and a nodal reward of 0 otherwise
        - 'linear': Give a nodal reward that linearly decreases from the middle of
          the allowed pressure interval, averaged over time
        - 'exponential': Give a maximum nodal reward of 1 at the middle of the
          allowed pressure interval with exponential reward decrease toward the
          extremes. Once again, average over time
        - 'linear+step': Superposition of the 'linear' and 'step' methods
        - 'exponential+step': Superposition of the 'exponential' and 'step' methods

        @returns The rewards averaged over nodes as a float. In case mod includes
        'step' and self.params.penalties.pressure_violation is positive, a penalty
        is added if pressure bounds were violated at any node.

        @note: This method is typically applied for pressures from s single time
        step in practise.
        '''
        if mod is None:
            mod = self.params.modifiers.pressure_mod
        else:
            self.params.modifiers.check_pressure_mod(mod)
        max_pressure = self.network_constraints.max_pressure
        min_pressure = self.network_constraints.min_pressure
        halfway_point = (max_pressure-min_pressure) / 2 + min_pressure

        if 'linear' in mod:
            nodal_rewards = (1 - np.abs(pressures - halfway_point)).mean(axis=0)

        elif 'exponential' in mod:
            max_reward = 1
            factor = 0.1
            exponent = (
                np.log(max_reward / factor)
                / (max_pressure-halfway_point)
            )
            nodal_rewards = (
                max_reward + factor
                - factor * np.exp(exponent*np.abs(pressures-halfway_point))
            ).mean(axis=0)
        else:
            nodal_rewards = np.ones(pressures.shape[1])

        if "step" in mod:
            # Check any violation across time
            violations = np.logical_or(
                pressures > self.network_constraints.max_pressure,
                pressures < self.network_constraints.min_pressure
            ).any(axis=0)
            nodal_rewards = nodal_rewards - violations
            # Check any violation across time and nodes
            if violations.any():
                nodal_rewards -= self.params.penalties.pressure_violation

        res = nodal_rewards.mean()
        if np.isnan(res):
            raise ValueError('Encountered NaN while calculating pressure objective')
        return float(res)

    def check_pump_and_tank_flow_sensors(self, sensor_config):
        required_sensor_locations = list(self.pumps) + list(self.tank_connections)
        for location in required_sensor_locations:
            if location not in sensor_config.flow_sensors:
                raise RuntimeError(
                    f"No flow sensor installed at link '{location}'. "
                    f"This is necessary to compute the tank flow objective."
                )

    def tank_objective(self, tank_flows, pump_flows):
        total_abs_tank_flow = np.abs(tank_flows).sum(axis=None)
        total_pump_flow = pump_flows.sum(axis=None)
        res = total_pump_flow / (total_pump_flow + total_abs_tank_flow)
        if np.isnan(res):
            raise ValueError('Encountered NaN while calculating tank flow objective')
        return float(res)

    def check_pump_efficiency_sensors(self, sensor_config):
        if not sensor_config.pump_efficiency_sensors:
            raise RuntimeWarning(
                f'No pump efficiency sensors installed. '
                f'The pump efficiency part of the objective will be set to zero.'
            )
            return False
        actual_sensors = sensor_config.pump_efficiency_sensors
        sensors_in_constraints = self.network_constraints.max_pump_efficiencies.index
        if not sorted(actual_sensors)==sorted(sensors_in_constraints):
            raise ValueError(
                f'Pump efficiency sensors and maximum pump efficiency values '
                f'in network constraints do not match:\n'
                f'actual sensors: {actual_sensors}\n'
                f'sensors implied by constraints: {sensors_in_constraints}'
            )
        else:
            return True

    def pump_efficiency_objective(self, pump_efficiencies, pump_efficiency_sensors):
        pump_efficiencies = pd.Series(
            pump_efficiencies.mean(axis=0),
            index=pump_efficiency_sensors
        )
        max_pump_efficiencies = self.network_constraints.max_pump_efficiencies
        normalized_pump_efficiencies = pump_efficiencies / max_pump_efficiencies
        res = normalized_pump_efficiencies.mean()
        if np.isnan(res):
            raise ValueError(
                f'Encountered NaN while calculating '
                f'pump efficiency objective'
            )
        return float(res)

    def pump_price_objective(self, pump_prices, pump_price_sensors):
        pump_prices = pd.Series(
            pump_prices.mean(axis=0),
            index=pump_price_sensors
        )
        if self.network_constraints.max_pump_prices is None:
            raise RuntimeError(
                f'max_pump_prices are not set in the network constraints. '
                f'They are required to compute the pump price objective.'
            )
        normalized_prices = pump_prices / self.network_constraints.max_pump_prices
        res = 1 - normalized_prices.mean()
        if np.isnan(res):
            raise ValueError(
                f'Encountered NaN whil computing '
                f'pump price objective'
            )
        return float(res)
        
    def smoothness_objective(self, last_three_speeds, max_pump_speed, mod=None):
        if mod is None:
            mod = self.params.modifiers.smoothness_mod
        else:
            self.params.modifiers.check_smoothness_mod(mod)
        res = None # make linter happy
        if mod=='zigzag':
            res = self.smoothness_objective_zigzag(last_three_speeds)
        elif mod=='curvature':
            res = self.smoothness_objective_curvature(
                last_three_speeds, max_pump_speed
            )
        _, speed, next_speed = last_three_speeds
        if (np.abs(speed-next_speed) > 0.25).any():
            res -= self.params.penalties.speed_change
        return res

    def smoothness_objective_zigzag(self, last_three_speeds):
        first, second, last = last_three_speeds
        zigzags = ((last-second) * (second-first) < -1e-8)
        smoothness_obj = float(1 - zigzags.sum() / len(self.pumps))
        return smoothness_obj

    def smoothness_objective_curvature(self, last_three_speeds, max_pump_speed):
        first, second, last = last_three_speeds
        curvatures = np.abs((last-second) - (second-first))
        normalized_curvatures = curvatures / max_pump_speed
        normalized_curvatures = np.clip(normalized_curvatures, a_min=None, a_max=1)
        smoothness_obj = 1 - normalized_curvatures.mean()
        return float(smoothness_obj)

    def full_objective(self, scada_data,
            speed_history=None, max_pump_speed=None,
            centered=True, return_all=False, verbose=False):
        """
        Compute the full objective.

        This consists of five parts:
        - pressure objective: Are the pressures within an acceptable range?
        - tank objective: Are the in- and outflows from/to tanks kept to a minimum?
        - pump efficiency objective: deprecated. Measures only wire-to-water
          efficiency. Rather use pump price objective (see below)
        - speed smoothness objective: Is the speed curve of the pumps smooth
          over time?
        - pump price objective: Are the prices for operating the pumps kept
          low?
        This function weights the results of the single objective functions by
        self.params.weights. Results are mostly between 0 (worst case) and 1 (best
        case) unless centered is True, in which case 0.5 is subtracted from the
        result to center it around zero.

        Exceptions to result bounds: If something other than 'step' is specified in
        self.modifiers.pressure_mod, the pressure objective may yield negative
        numbers as outcome (see the documentation of pressure_objective). The
        pump price objective may also yield negative numbers due to empirical
        normalization.

        @param scada_data, epyt_flow.simulation.scada_data.ScadaData
        sensor readings

        @param speed_history, iterable, the last three pump speeds
        defaults to None, this is only needed for the smoothness objective
        (i.e. it can be None if the corresponding weight is zero)

        @param max_pump_speed, float, default=None
        needed for normalization of the smoothness objective

        @param centered, bool, default=True
        If 0.5 should be subtracted from the reward to center it around zero

        @param verbose, bool, If True, print the single objectives

        @returns the weighted result of all objectives
        """
        pressure_obj = self.pressure_objective(scada_data.get_data_pressures())

        sensor_config = scada_data.sensor_config
        self.check_pump_and_tank_flow_sensors(sensor_config)
        tank_flows = scada_data.get_data_flows(
            sensor_locations=self.tank_connections
        )
        pump_flows = scada_data.get_data_flows(sensor_locations=self.pumps)
        tank_obj = self.tank_objective(tank_flows, pump_flows)

        need_efficiency = (
            self.check_pump_efficiency_sensors(sensor_config)
            and
            (self.params.weights.pump_efficiency > 0)
        )
        if need_efficiency:
            pump_efficiencies = scada_data.get_data_pumps_efficiency()
            pump_efficiency_sensors = sensor_config.pump_efficiency_sensors
            pump_efficiency_obj = self.pump_efficiency_objective(
                pump_efficiencies, pump_efficiency_sensors
            )
        else:
            pump_efficiency_obj = 0
        if self.params.weights.speed_smoothness > 0:
            if speed_history is None or max_pump_speed is None:
                raise ValueError(
                    f'speed_history and max_pump_speed are required '
                    f'to compute the smoothness objective.'
                )
            smoothness_obj = self.smoothness_objective(speed_history, max_pump_speed)
        else:
            smoothness_obj = 0
        if self.params.weights.pump_price > 0:
            if self.network_constraints.max_pump_prices is None:
                raise ValueError(
                    f'max_pump_prices must be given in network_constraints '
                    f'to compute the pump price objective'
                )
            pump_prices = scada_data.get_data_pumps_energyconsumption()
            pump_price_sensors = sensor_config.pump_energyconsumption_sensors
            pump_price_obj = self.pump_price_objective(
                pump_prices, pump_price_sensors
            )
        else:
            pump_price_obj = 0
        weighted_result = (
            self.params.weights.pressure_violation * pressure_obj
            + self.params.weights.abs_tank_flow * tank_obj
            + self.params.weights.pump_efficiency * pump_efficiency_obj
            + self.params.weights.speed_smoothness * smoothness_obj
            + self.params.weights.pump_price * pump_price_obj
        )
        if centered:
            weighted_result -= 0.5
        hydraulically_unstable = (
            scada_data.warnings_code[0] == W_HYDRAULICALLY_UNSTABLE
        )
        if hydraulically_unstable:
            weighted_result -= self.params.penalties.hydraulic_instability
        if verbose:
            print(f'pressure objective: {pressure_obj:.4f}')
            print(f'tank objective: {tank_obj:.4f}')
            print(f'pump efficiency objective: {pump_efficiency_obj:.4f}')
            print(f'smoothness objective: {smoothness_obj:.4f}')
            print(f'pump price objective: {pump_price_obj:.4f}')
            print(f'full objective: {weighted_result:.4f}')
        if return_all:
            res = dict(
                pressure_obj=pressure_obj,
                tank_obj=tank_obj,
                pump_efficiency_obj=pump_efficiency_obj,
                smoothness_obj=smoothness_obj,
                pump_price_obj=pump_price_obj,
                hydraulically_unstable=hydraulically_unstable,
                reward=weighted_result
            )
            return res
        else:
            return weighted_result

    def full_objective_stepwise(self, scadas, return_all=False, **kwargs):
        results = dict()
        for scada in scadas:
            result = self.full_objective(scada, return_all=return_all, **kwargs)
            time = scada.sensor_readings_time[0]
            results[time] = result
        if return_all:
            return pd.DataFrame(results).T
        else:
            return pd.Series(results)

    def compare2standard_optimizer(self):
        raise NotImplementedError('Not implemented yet.')

