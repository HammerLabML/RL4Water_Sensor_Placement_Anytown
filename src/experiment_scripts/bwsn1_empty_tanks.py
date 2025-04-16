import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from epyt_flow.simulation import ScenarioSimulator
import pandas as pd
import matplotlib.pyplot as plt

inp_file = '../Data/BWSN1/bwsn1.inp'
sim = ScenarioSimulator(f_inp_in=inp_file)
sim.set_general_parameters(flow_units_id=5)
topo = sim.get_topology()
junctions = topo.get_all_junctions()
sim.set_pressure_sensors(junctions)
sim.epanet_api.setNodeTankInitialLevel([0., 0.])
scada = sim.run_hydraulic_simulation()
pres = pd.DataFrame(
    scada.get_data_pressures(),
    index=range(0, sim.get_simulation_duration() + 1, 3600),
    columns=junctions
)
mean_pres = pres.mean(axis=0).sort_values()
fig, ax = plt.subplots()
mean_pres.plot(ax=ax)
# Max and Min pressure levels
ax.axhline(28)
ax.axhline(70)
plt.show()
