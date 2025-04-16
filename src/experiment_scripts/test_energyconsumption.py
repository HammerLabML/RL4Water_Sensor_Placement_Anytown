import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

import matplotlib.pyplot as plt

import sim_util

inp_file = '../Data/Anytown/ATM-v3.inp'
sim = sim_util.standard_sim(inp_file)
sim = sim_util.set_pump_speeds(sim, ['b1', 'b2', 'b3'], [1.0, 1e-10, 1e-10], 0)
scada = sim.run_hydraulic_simulation()
sim.close()
ec = scada.get_data_pumps_energyconsumption()
plt.plot(range(ec.shape[0]), ec[:, 0])
plt.show()

