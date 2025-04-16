import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from os import path
from epyt_flow.simulation import ScenarioSimulator
import sim_util
from objective_calculator import NetworkConstraints

network_dir = '../Data/Net3'
network_file = path.join(network_dir, 'net3.inp')
network_constraints_file = path.join(network_dir, 'net3_constraints.yaml')
network_constraints = NetworkConstraints.from_yaml(network_constraints_file)
sim = sim_util.standard_sim(network_file)
max_pump_speed = 1
network_constraints.max_pump_prices = sim_util.max_pump_prices(sim, max_pump_speed)
network_constraints.max_abs_flows = sim_util.max_abs_flows(sim, max_pump_speed)
network_constraints.to_yaml(network_constraints_file)
sim.close()

