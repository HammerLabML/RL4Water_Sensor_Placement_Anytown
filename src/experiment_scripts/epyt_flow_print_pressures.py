from epyt_flow.simulation import ScenarioSimulator

inp_file = 'net1.inp'
sim = ScenarioSimulator(f_inp_in=inp_file)
sim.place_pressure_sensors_everywhere()
scada = sim.run_hydraulic_simulation()
pressures = scada.get_data_pressures()
print('Computed Pressures using EPyT-Flow')
print(pressures[:10, :])

