import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from os import path
from sim_util import scenario_generator_factory

data_dir = '../Data/Net1'
scenario_files = [
    path.join(data_dir, scenario_file)\
    for scenario_file in ['net1.inp', 'net1_half_filled_tank.inp']
]
gen = scenario_generator_factory(scenario_files)
print('Calling gen the first time')
sc, sim = next(gen)
print('Initial Tank Levels')
print(sim.epanet_api.getNodeTankInitialLevel())
print('Calling gen the second time')
sc, sim = next(gen)
print('Initial Tank Levels')
print(sim.epanet_api.getNodeTankInitialLevel())
