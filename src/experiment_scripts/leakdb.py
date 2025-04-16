import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from os import path

from pump_control_envs import leakdb_env
import sim_util

data_dir = '../Data/Net1/LeakDB_Versions'
network_files = [path.join(data_dir, f'Scenario-{i}.inp') for i in range(1, 1001)]
sim1 = sim_util.standard_sim(network_files[0])
sim1 = sim_util.set_pump_speeds(sim1, ['9'], [0.5], 0)
scada1 = sim1.run_hydraulic_simulation()
print(scada1.get_data_pressures()[0,:].mean())

sim2 = sim_util.standard_sim(network_files[0])
sim2 = sim_util.set_pump_speeds(sim2, ['9'], [1.5], 0)
scada2 = sim2.run_hydraulic_simulation()
print(scada2.get_data_pressures()[0,:].mean())
