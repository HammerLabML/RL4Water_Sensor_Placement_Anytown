import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from pump_control_envs import leakdb_env
from normalize_hydraulics import NormalizeHydraulics

objective_weights_file = '../Data/Parameters_for_Optimization/objective_weights_5.yaml'
env1 = leakdb_env(scenario_nrs=[1], max_pump_speed=1, objective_weights_file=objective_weights_file)
env1 = NormalizeHydraulics(env1)

env2 = leakdb_env(scenario_nrs=[302], max_pump_speed=1, objective_weights_file=objective_weights_file)
env2 = NormalizeHydraulics(env2)

print(f'Env1 max flows: {env1.nonzero_max_abs_flows}')
print(f'Env2 max_flows: {env2.nonzero_max_abs_flows}')
print(f'Env1 max prices: {env1.max_pump_prices}')
print(f'Env2 max prices: {env2.max_pump_prices}')
env1.close()
env2.close()

