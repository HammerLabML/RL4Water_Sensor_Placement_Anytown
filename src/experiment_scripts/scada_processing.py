import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

import sim_util
from os import path
from objective_calculator import NetworkConstraints
import yaml

data_dir = '../Data/Anytown'
inp_file = path.join(data_dir, 'anytown.inp')
network_constraints_file = path.join(data_dir, 'anytown_constraints.yaml')
network_constraints = NetworkConstraints.from_yaml(network_constraints_file)
sim = sim_util.standard_sim(inp_file)
initial_pump_speeds_file = path.join(data_dir, 'initial_pump_speeds.yaml')
with open(initial_pump_speeds_file, 'r') as fp:
    initial_pump_speeds = yaml.safe_load(fp)
sim = sim_util.set_pump_speeds_from_dict(
    sim, initial_pump_speeds, time=0
)
scada = sim.run_hydraulic_simulation()
orig_data = scada.get_data()
print(orig_data[0])
normalized_data = sim_util.constraint_based_normalization(scada, network_constraints)
print(normalized_data[0])

