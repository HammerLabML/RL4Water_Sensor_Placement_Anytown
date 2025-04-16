from objective_calculator import ObjectiveCalculator
from objective_calculator import NetworkConstraints
from scipy.optimize import minimize, Bounds
import numpy as np
import argparse
import sim_util
import yaml
import pandas as pd
import file_util

from my_scenario_simulator import MyScenarioSimulator

class SequentialPumpSpeedFinder():

    '''
    Class to find the initial pump speed to guide the RL agent.

    @param sim: my_scenario_simulator.MyScenarioSimulator
    should have the following sensors:
    - pressure sensors everywhere
    - pump efficiency sensors everywhere
    - flow sensors at pumps and connections to tanks
    should have a short simulation duration (60s)
    
    @param objective_calculator: ObjectiveCalculator

    @param optimizer_params: dict, must contain the following fields:
    - max_pump_speed
    - method
    - maxiter
    May also contain an optional field n_runs for multiple runs of the optimizer.
    '''
    def __init__(self, sim, objective_calculator, optimizer_params):
        self.sim = sim
        self.sim.epanet_api.deleteControls()
        self.objective_calculator = objective_calculator
        self.optimizer_params = optimizer_params
        # Store bounds, options and number of pumps for optimizers
        self._optimization_constants = self._derive_optimization_constants()
        self._next_time_step = 0
        self._reporting_time_step = self.sim.get_reporting_time_step()
        self._last_time_step = self.sim.get_simulation_duration()
        self.best_pump_speeds = self._construct_empty_speeds_df()
        # A duration of 60 seconds is enough to optimize the speed
        # at the first time step.
        self.sim.set_general_parameters(simulation_duration=60)

    def _construct_empty_speeds_df(self):
        all_time_steps = range(0, self._last_time_step, self._reporting_time_step)
        df = pd.DataFrame(
            index=all_time_steps,
            columns=self.objective_calculator.pumps
        )
        df.index.name = 'Time'
        return df

    def _derive_optimization_constants(self):
        n_pumps = len(self.objective_calculator.pumps)
        bounds = Bounds(
            lb=np.ones(n_pumps) * 1e-4,
            ub=np.ones(n_pumps) * self.optimizer_params['max_pump_speed']
        )
        options = dict(maxiter=self.optimizer_params['maxiter'], disp=True)
        optimization_constants = dict()
        optimization_constants['n_pumps'] = n_pumps
        optimization_constants['bounds'] = bounds
        optimization_constants['options'] = options
        return optimization_constants

    def negative_objective(self, next_pump_speeds):
        # First, set all previously optimal speeds for previous timesteps
        for time, speeds_dict in self.best_pump_speeds.iterrows():
            if speeds_dict.hasnans:
                break
            self.sim = sim_util.set_pump_speeds_from_dict(
                self.sim, speeds_dict, time
            )
        # Now, try a set of pump speeds for the next timestep
        self.sim = sim_util.set_pump_speeds(
            self.sim, self.objective_calculator.pumps,
            next_pump_speeds, self._next_time_step
        )
        all_scadas = list(self.sim.run_hydraulic_simulation_as_generator())
        is_current = lambda sc: sc.sensor_readings_time[0]==self._next_time_step
        current_scada = list(filter(is_current, all_scadas))[0]
        objective = self.objective_calculator.full_objective(current_scada)
        return (-1) * objective

    def prepare_next_iteration(self):
        self.sim.remove_actuator_events()
        self._next_time_step += self._reporting_time_step
        current_simulation_duration = self.sim.get_simulation_duration()
        current_simulation_duration += self._reporting_time_step
        self.sim.set_general_parameters(
            simulation_duration=current_simulation_duration
        )

    def print_speeds(self, speeds):
            for pump, speed in speeds.items():
                print(f'Pump {pump}: {speed:.4f}')
            print('='*30)
            print()

    def compute_initial_guess(self, use_previous=True, multiple_runs=False):
        if use_previous:
            previous_time_step = self._next_time_step - self._reporting_time_step
            x0 = self.best_pump_speeds.loc[previous_time_step, :].to_numpy()
        else:
            n_pumps = self._optimization_constants['n_pumps']
            bounds = self._optimization_constants['bounds']
            max_pump_speed = self.optimizer_params['max_pump_speed']
            if multiple_runs:
                x0 = np.random.uniform(low=bounds.lb, high=bounds.ub, size=n_pumps)
            else:
                # make sure to start somewhere in the middle
                x0 = np.ones(n_pumps) * (max_pump_speed / 2)
                max_offset = max_pump_speed / 10
                x0 += np.random.uniform(
                    low=-max_offset, high=max_offset, size=n_pumps
                )
        return x0

    def find_only_initial_pump_speeds(self, n_runs=1, output_file=None):
        if 'n_runs' in self.optimizer_params.keys():
            n_runs = self.optimizer_params['n_runs']
        if n_runs==1:
            x0 = self.compute_initial_guess(use_previous=False, multiple_runs=False)
            best_res = minimize(
                fun=self.negative_objective,
                x0=x0,
                method=self.optimizer_params['method'],
                options=self._optimization_constants['options'],
                bounds=self._optimization_constants['bounds']
            )
        else:
            best_res = None
            for run in range(n_runs):
                if run%2!=0 or run==n_runs-1:
                    print(f'Iteration {run+1}')
                x0 = self.compute_initial_guess(
                    use_previous=False, multiple_runs=True
                )
                res = minimize(
                    fun=self.negative_objective,
                    x0=x0,
                    method=self.optimizer_params['method'],
                    options=self._optimization_constants['options'],
                    bounds=self._optimization_constants['bounds']
                )
                if best_res is None or res.fun < best_res.fun:
                    best_res = res
        found_pump_speeds = dict(
            zip(self.objective_calculator.pumps, best_res.x, strict=True)
        )
        print('Optimization completed')
        self.print_speeds(found_pump_speeds)
        all_results = self.print_objectives_from_result(best_res)
        pressure_obj, tank_obj, pump_efficiency_obj, total_obj = all_results
        if output_file is not None:
            self.save_results_for_initial_time_step(
                pressure_obj, tank_obj, pump_efficiency_obj, total_obj,
                best_res, output_file
            )
        self.sim.close()

    def find_next_pump_speeds(self):
        x0 = self.compute_initial_guess(
            use_previous=(self._next_time_step!=0), multiple_runs=False
        )
        best_res = minimize(
            fun=self.negative_objective,
            x0=x0,
            method=self.optimizer_params['method'],
            options=self._optimization_constants['options'],
            bounds=self._optimization_constants['bounds']
        )
        found_pump_speeds = dict(
            zip(self.objective_calculator.pumps, best_res.x, strict=True)
        )
        self.best_pump_speeds.loc[self._next_time_step, :] = found_pump_speeds
        print(f'Optimization at {self._next_time_step} completed')
        self.print_speeds(found_pump_speeds)
        self.print_objectives_from_result(best_res)
        self.prepare_next_iteration()

    def find_all_pump_speeds(self, param_log_file=None, result_file=None):
        for output_file in [param_log_file, result_file]:
            if output_file is not None:
                file_util.check_writability(output_file)
        while self._next_time_step <= self._last_time_step:
            self.find_next_pump_speeds()
        if param_log_file is not None:
            log = self.to_dict()
            with open(param_log_file, 'w') as fp:
                yaml.dump(log, fp)
        if result_file is not None:
            self.best_pump_speeds.to_csv(result_file)
    
    def print_objectives_from_result(self, optimization_result, time=None):
        if time is None:
            time = self._next_time_step
        self.sim = sim_util.set_pump_speeds(
            self.sim, self.objective_calculator.pumps, optimization_result.x,
            time
        )
        all_scadas = list(self.sim.run_hydraulic_simulation_as_generator())
        # Pick only the results from the time step
        # for which the optimization was actually performed
        scada = [sc for sc in all_scadas if sc.sensor_readings_time[0]==time][0]
        print(f'Result of the objective function at time {time}')
        all_results = self.objective_calculator.full_objective(
            scada, return_all=True, verbose=True
        )
        return all_results

    def get_excluded_nodes(self):
        '''Get nodes without pressure sensors. Typically water sources'''
        all_nodes = self.sim.get_topology().get_all_nodes()
        nodes_with_sensors = self.sim.sensor_config.pressure_sensors
        excluded_nodes = [n for n in all_nodes if n not in nodes_with_sensors]
        return excluded_nodes

    def to_dict(self):
        log = {}
        log['network_file'] = self.sim.f_inp_in
        log['objective_params'] = self.objective_calculator.params.to_dict()
        log['optimizer_params'] = self.optimizer_params
        log['excluded_nodes'] = self.get_excluded_nodes()
        tanks = self.sim.get_topology().get_all_tanks()
        initial_tank_levels = self.sim.epanet_api.getNodeTankInitialLevel()
        # Conversion to normal floats is necessary for proper logging
        initial_tank_levels = list(map(float, initial_tank_levels))
        log['initial_tank_levels'] = dict(
            zip(tanks, initial_tank_levels, strict=True)
        )
        return log

    def save_results_for_initial_time_step(self, pressure_obj, tank_obj,
            pump_efficiency_obj, total_obj,
            optimization_result, output_file):
        log = self.to_dict()
        pump_speeds = list(map(float, optimization_result.x))
        log['pump_speeds'] = (
            dict(zip(self.objective_calculator.pumps, pump_speeds, strict=True))
        )
        log['pressure_objective'] = float(pressure_obj)
        log['tank_objective'] = float(tank_obj)
        log['pump_efficiency_objective'] = float(pump_efficiency_obj)
        log['full_objective'] = float(total_obj)
        with open(output_file, 'w') as fp:
            yaml.dump(log, fp)

def parse_command_line():
    parser = argparse.ArgumentParser(description='Initial Pump Speed Optimizer')
    parser.add_argument('network_file', type=str, help='Water network .inp-file')
    network_constraints_help = (
        f'A .yaml-file containing the following fields:\n'
        f'min_pressure, max_pressure, max_pump_efficiencies. '
        f'Write values for each pump in the following lines in the form '
        f'<pump_id>: <max_efficiency> and indent by one level'
    )
    parser.add_argument(
        'network_constraints_file', type=str, help=network_constraints_help
    )
    objective_weights_help = (
        f'.yaml file containing the weights for the objective function. '
        f'The following fields are required: '
        f'pressure_violation, abs_tank_flow, pump_efficiency'
    )
    parser.add_argument(
        'objective_weights_file', type=str, help=objective_weights_help
    )
    optimizer_params_help = (
        f'.yaml-file containing parameters for the optimizer. These must include: '
        f'max_pump_speed (used as upper bound), method (e.g. "Nelder-Mead" or '
        f'"Powell", see scipy.optimize.minimize), maxiter (number of optimizer '
        f'iterations). In addition to these, n_runs may be specified if only the '
        f'speed for the initial timestep should be computed (see --only-initial). '
        f'The optimization will then be repeated n_runs times from different '
        f'starting points and the best result will be taken.'
    )
    parser.add_argument(
        'optimizer_params_file', type=str,
        help=optimizer_params_help
    )
    min_tank_levels_help=(
            f'If this is given, initial tank levels are set to the'
            f' smallest possible value'
        )
    parser.add_argument(
        '--set_min_tank_levels', action='store_true',
        help=min_tank_levels_help
    )
    only_initial_help = (
        f'Compute only the optimal pump speeds for the initial timestep. '
        f'This is useful as a starting point for an RL agent taking actions '
        f'relative to the current pump speed (e.g. increasing or decreasing) '
        f'it by a fixed amount). You may specify n_runs in the optimizer_params_file'
        f' to run the optimization n_runs times from different starting points and '
        f'report the best result.'
    )
    parser.add_argument(
        '--only_initial', action='store_true', help=only_initial_help
    )
    param_log_file_help = (
       f'IMPORTANT: Only used if the only_initial option is NOT set. '
       f'log optimization parameters to the given file in YAML format' 
    )
    parser.add_argument(
        '-p', '--param_log_file', type=str,
        help=param_log_file_help
    )
    result_file_help = (
        f'IMPORTANT: Only used if the only_initial option is NOT set. '
        f'write results to the given file in .csv-format, indexed by time'
    )
    parser.add_argument(
        '-r', '--result_file', type=str,
        help=result_file_help
    )
    output_file_help = (
        f'IMPORTANT: Only used if the --only_initial option is present. '
        f'log parameters and results to the given file in YAML format.'
    )
    parser.add_argument(
        '-o', '--output_file', type=str,
        help=output_file_help
    )
    args = parser.parse_args()
    return args

def setup_speed_finder(args):
    initial_pump_speed_finder = _setup_speed_finder(
        args.network_file,
        args.network_constraints_file,
        args.objective_weights_file,
        args.optimizer_params_file,
        set_min_tank_levels=args.set_min_tank_levels,
    )
    return initial_pump_speed_finder

def _setup_speed_finder(network_file, network_constraints_file,
        objective_weights_file, optimizer_params_file, set_min_tank_levels=False):
    objective_calculator = ObjectiveCalculator.from_files(
        network_constraints_file, objective_weights_file, network_file
    )
    sim = sim_util.standard_sim(network_file)
    if set_min_tank_levels:
        sim = sim_util.set_tank_levels2min(sim)
    with open(optimizer_params_file, 'r') as fp:
        optimizer_params = yaml.safe_load(fp)
    sequential_pump_speed_finder = SequentialPumpSpeedFinder(
        sim, objective_calculator, optimizer_params
    )
    return sequential_pump_speed_finder

if __name__=='__main__':
    args = parse_command_line()
    sequential_pump_speed_finder = setup_speed_finder(args)
    if args.only_initial:
        sequential_pump_speed_finder.find_only_initial_pump_speeds(
            output_file=args.output_file
        )
    else:
        sequential_pump_speed_finder.find_all_pump_speeds(
            param_log_file=args.param_log_file, result_file=args.result_file
        )

