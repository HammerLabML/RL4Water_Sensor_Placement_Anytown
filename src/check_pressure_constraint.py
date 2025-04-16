from epyt_flow.simulation import ScenarioSimulator
from objective_calculator import NetworkConstraints
import sim_util
import yaml
import pandas as pd
from os import path

def check_pressures(pressures, network_constraints, pressure_sensors):
    '''Return those sensors for which pressures violated the constraints'''
    pressures = pd.DataFrame(pressures, columns=pressure_sensors)
    did_exceed = (
        (pressures > network_constraints.max_pressure)
        | (pressures < network_constraints.min_pressure)
    ).any(axis=0)
    return pressures.loc[:, did_exceed]

def get_violations_for_network(network_file, network_constraints_file, pump_speeds_file):
    sim = sim_util.standard_sim(network_file, duration=60)
    network_constraints = NetworkConstraints.from_yaml(network_constraints_file)
    with open(pump_speeds_file, 'r') as fp:
        file_dict = yaml.safe_load(fp)
    initial_pump_speeds = file_dict['pump_speeds']
    sim = sim_util.set_initial_pump_speeds_from_dict(sim, initial_pump_speeds)
    scada_data = sim.run_hydraulic_simulation()
    pressures = scada_data.get_data_pressures()
    pressure_sensors = sim.sensor_config.pressure_sensors
    sim.close()
    return check_pressures(pressures, network_constraints, pressure_sensors)

if __name__=='__main__':
    network_dir = '../Data/Net1/'
    network_file = path.join(network_dir, 'net1.inp')
    network_constraints_file = path.join(network_dir, 'net1_constraints.yaml')
    pump_speeds_file = '../Results/Initial_Pump_Speeds/Net1/output_1.yaml'
    res = get_violations_for_network(
        network_file, network_constraints_file, pump_speeds_file
    )
