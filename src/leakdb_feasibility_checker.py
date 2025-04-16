from os import path
import numpy as np
import yaml

import sim_util
from pump_control_envs import leakdb_env
from objective_calculator import ObjectiveCalculator

data_dir = '../Data/Net1/LeakDB_Versions'
network_files = [path.join(data_dir, f'Scenario-{i}.inp') for i in range(1, 1001)]
feasible = {network_file: None for network_file in network_files}
network_constraints_file = '../Data/Net1/net1_constraints.yaml'
objective_weights_file = '../Data/Parameters_for_Optimization/objective_weights_1.yaml'
my_objective_calculator = ObjectiveCalculator.from_files(
    network_constraints_file, objective_weights_file, network_files[0]
)
for i, network_file in enumerate(network_files):
    if (i+1)%10==0:
        print(f'Processing {i+1}th file...')
    sim = sim_util.standard_sim(network_file, duration=60)
    sim = sim_util.set_tank_levels2min(sim)
    for speed in np.arange(0.05, 1.05, 0.05)[::-1]: # reversed order
        sim = sim_util.set_pump_speeds(sim, ['9'], [speed], 0)
        scadas = list(sim.run_hydraulic_simulation_as_generator())
        initial_scada = scadas[0]
        pressures = initial_scada.get_data_pressures()
        pressure_obj = my_objective_calculator.pressure_objective(pressures)
        if pressure_obj==1:
            feasible[network_file] = round(float(speed), ndigits=2)
            break
    sim.close()
    if feasible[network_file] is None:
        feasible[network_file] = False
output_file = '../Data/Net1/LeakDB_Versions/max_feasibility.yaml'
with open(output_file, 'w') as fp:
    yaml.dump(feasible, fp)

