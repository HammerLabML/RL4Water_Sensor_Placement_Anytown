import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from epyt_flow.simulation import ScenarioSimulator
import sim_util

inp_file = '../Data/Net1/net1.inp'
sim = sim_util.standard_sim(inp_file)
times = range(0, 24*3600+1, 300)
for time in times:
    sim = sim_util.set_pump_speeds(sim, ['9'], [0.45], time)
scada = sim.run_hydraulic_simulation()

