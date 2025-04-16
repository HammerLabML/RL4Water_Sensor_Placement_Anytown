import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

import matplotlib.pyplot as plt
import sim_util

def plot_speeds_to_efficiency():
    inp_file = '../Data/Anytown/ATM-v3.inp'
    sim = sim_util.standard_sim(inp_file)
    pump = sim.get_topology().get_all_pumps()[0]
    speeds = [i/10 for i in range(1, 10)]
    speeds.extend(list(range(1, 10)))
    efficiencies = []
    sim.set_general_parameters(simulation_duration=60)
    sim.set_pump_efficiency_sensors([pump])
    for speed in speeds:
        sim = sim_util.set_pump_speeds(sim, [pump], [float(speed)], time=0)
        scada = sim.run_hydraulic_simulation()
        efficiencies.append(scada.get_data_pumps_efficiency().flatten()[0])
        sim.remove_actuator_events()
    sim.close()
    plt.plot(speeds, efficiencies)
    plt.show()

plot_speeds_to_efficiency()
