import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

import sim_util

inp_file = '../Data/Anytown/ATM.inp'
sim = sim_util.standard_sim(inp_file, delete_controls=True)
scada1 = sim.run_hydraulic_simulation()
scada2 = sim.run_hydraulic_simulation()
print((scada1==scada2).all(axis=None))

