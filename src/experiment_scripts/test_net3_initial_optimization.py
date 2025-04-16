import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from epyt_flow.simulation import ScenarioSimulator
from sim_util import set_initial_pump_speeds
import numpy as np

def net3_sim():
    inp_file = '../Data/Net3/net3.inp'
    sim = ScenarioSimulator(f_inp_in=inp_file)
    sim.set_general_parameters(simulation_duration=1, flow_units_id=5)
    sim.place_pressure_sensors_everywhere()
    return sim

pumps = ['10', '335']
sim = net3_sim()
speeds1 = [2.76, 0.992]
set_initial_pump_speeds(sim, pumps, speeds1)
scada = sim.run_hydraulic_simulation()
pres1 = scada.get_data_pressures()

sim = net3_sim()
speeds2 = [5.0, 0.992]
set_initial_pump_speeds(sim, pumps, speeds2)
scada = sim.run_hydraulic_simulation()
pres2 = scada.get_data_pressures()

diff = np.abs(pres1 - pres2)
print(diff)
