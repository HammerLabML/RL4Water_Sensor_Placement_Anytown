from epyt_flow.simulation import ScenarioSimulator as SIM

sim = SIM(f_inp_in='net1.inp')
sim.set_general_parameters(simulation_duration=60)
gen = sim.run_hydraulic_simulation_as_generator()
scadas = [scada for scada in gen]
# Expected Behavior: Get only one scada object with a sensor readings time of 0
print([scada.sensor_readings_time for scada in scadas])
sim.close()

