from epyt_flow.simulation import ModelUncertainty, ScenarioSimulator
from epyt_flow.uncertainty import PercentageDeviationUncertainty
import matplotlib.pyplot as plt
import pandas as pd

demand_pattern_uncertainty = PercentageDeviationUncertainty(deviation_percentage=0.05)
model_uncertainty = ModelUncertainty(
    demand_pattern_uncertainty=demand_pattern_uncertainty
)

def uncertain_demands(network_file):
    sim = ScenarioSimulator(network_file)
    sim.set_model_uncertainty(model_uncertainty)
    sim.set_demand_sensors(sim.get_topology().get_all_junctions()[:1])
    scada = sim.run_hydraulic_simulation()
    demands = scada.get_data_demands().flatten()
    sim_time = scada.sensor_readings_time
    demands = pd.Series(demands, index=sim_time)
    return demands

network_file = '../../Data/Anytown/ATM.inp'
demands_1 = uncertain_demands(network_file)
demands_2 = uncertain_demands(network_file)
fig, ax = plt.subplots()
ax.plot(demands_1.index, demands_1, label='demands_1')
ax.plot(demands_2.index, demands_2, label='demands_2')
plt.legend()
plt.show()
