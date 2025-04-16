from epyt_flow.simulation import ScenarioSimulator
from epyt_flow.simulation.scada import ScadaDataExport as SDE

sim = ScenarioSimulator(f_inp_in='net1.inp')
# This should not change anything, but just to be certain...
sim.set_general_parameters(flow_units_id=5) # liters/sec, SI unit
sim.place_pressure_sensors_everywhere()
scada = sim.run_hydraulic_simulation()
sde = SDE(f_out='/dev/null')
print(sde.create_column_desc(scada))
