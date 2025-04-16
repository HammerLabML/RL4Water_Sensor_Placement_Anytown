import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from objective_calculator import NetworkConstraints, ObjectiveWeights, ObjectiveCalculator
from epyt_flow.simulation import ScenarioSimulator
from epyt_flow.simulation.events.actuator_events import PumpSpeedEvent

network_constraints = NetworkConstraints.from_yaml('../Data/Anytown/anytown_constraints.yaml')
objective_weights = ObjectiveWeights.from_dict(
    {'pressure_violation': 0.9, 'abs_tank_flow': 0, 'pump_efficiency': 0.1}
)
pumps = ['78', '79', '80']
tank_connections = ['142', '143']
objective_calculator = ObjectiveCalculator(
    network_constraints, objective_weights, pumps, tank_connections
)

inp_file = '../Data/Anytown/anytown.inp'
sim = ScenarioSimulator(f_inp_in=inp_file)
sim.place_pressure_sensors_everywhere()
sim.set_flow_sensors(sensor_locations=(pumps+tank_connections))
sim.place_pump_efficiency_sensors_everywhere()
sim.set_general_parameters(simulation_duration=60)
for pump in pumps:
    pump_speed_event = PumpSpeedEvent(pump_speed=2., pump_id=pump, time=0)
    sim.add_actuator_event(pump_speed_event)
scada_data = sim.run_hydraulic_simulation()
sim.close()
objective = objective_calculator.full_objective(scada_data, verbose=True)
