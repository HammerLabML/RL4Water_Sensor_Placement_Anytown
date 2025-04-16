from epyt_flow.simulation import ScenarioSimulator
import sim_util
from objective_calculator import ObjectiveCalculator, NetworkConstraints
import pandas as pd
import matplotlib.pyplot as plt
from os import path

inp_file = '../Data/Net1/net1_with_controls.inp'
sim = sim_util.standard_sim(inp_file, delete_controls=False)

network_constraints_file = '../Data/Net1/net1_constraints.yaml'
objective_weights_file = '../Data/Parameters_for_Optimization/objective_weights_1.yaml'
objective_calculator = ObjectiveCalculator.from_files(
    network_constraints_file, objective_weights_file, inp_file
)

gen = sim.run_hydraulic_simulation_as_generator()
rewards = objective_calculator.full_objective_stepwise(gen)
sim.close()

results_dir = '../Results/No_Uncertainty/Net1/Lower_Baseline'
objective_calculator.to_yaml(path.join(results_dir,'lower_baseline_calculator.yaml'))
rewards.to_csv(path.join(results_dir, 'rewards_lower_baseline.csv'))
fig, ax = plt.subplots()
rewards.plot(ax=ax)
plt.savefig(path.join(results_dir, 'lower_baseline.png'))
plt.show()

