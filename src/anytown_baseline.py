import sim_util
from os import path
from objective_calculator import ObjectiveCalculator
import yaml
import pandas as pd
import matplotlib.pyplot as plt

data_dir = '../Data/Anytown'
network_file = path.join(data_dir, 'anytown.inp')
sim = sim_util.standard_sim(network_file)
network_constraints_file = path.join(data_dir, 'anytown_constraints.yaml')
objective_weights_file = '../Data/Parameters_for_Optimization/objective_weights_1.yaml'
objective_calculator = ObjectiveCalculator.from_files(
    network_constraints_file, objective_weights_file, network_file
)
time_steps = range(0, sim.get_simulation_duration()+1, sim.get_reporting_time_step())
pump_speeds_file = path.join(data_dir, 'initial_pump_speeds.yaml')
with open(pump_speeds_file, 'r') as fp:
    pump_speeds = yaml.safe_load(fp)
for time_step in time_steps:
    sim = sim_util.set_pump_speeds_from_dict(sim, pump_speeds, time_step)

gen = sim.run_hydraulic_simulation_as_generator()
rewards = objective_calculator.full_objective_stepwise(gen)
sim.close()

results_dir = '../Results/No_Uncertainty/Anytown/Lower_Baseline'
sim_util.save_lower_baseline(results_dir, rewards, objective_calculator, show=True)

