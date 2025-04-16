from my_scenario_simulator import MyScenarioSimulator
from epyt_flow.simulation.events.actuator_events import PumpSpeedEvent
from epyt_flow.simulation.scada import ScadaDataExport
import numpy as np
import pandas as pd
from os import path
import matplotlib.pyplot as plt

def standard_sim(inp_file, duration=None, delete_controls=True,
        add_pump_state_sensors=False, exclude_non_demand_nodes=False):
    sim = MyScenarioSimulator(f_inp_in=inp_file)
    # Set flow units to liters per second, ensuring that the metric system is used
    sim.set_general_parameters(flow_units_id=5)
    topology = sim.get_topology()

    junctions = topology.get_all_junctions()
    if exclude_non_demand_nodes:
        for junc in junctions:
            base = sim.get_node_base_demand(junc)
            if base < 1e-3:
                junctions.remove(junc)
    sim.set_pressure_sensors(junctions)

    sim.place_pump_efficiency_sensors_everywhere()
    sim.place_pump_energyconsumption_sensors_everywhere()
    sim.set_tank_sensors(topology.get_all_tanks())
    if add_pump_state_sensors:
        sim.place_pump_state_sensors_everywhere()
    pumps = topology.get_all_pumps()
    tank_connections = get_tank_connections(topology)
    # I used to do the following with a union of sets
    # which introduced a really nasty bug by randomly messing up the order of sensors
    flow_sensor_locations = pumps + [t for t in tank_connections if t not in pumps]
    sim.set_flow_sensors(sensor_locations=flow_sensor_locations)
    if duration is not None:
        sim.set_general_parameters(simulation_duration=duration)
    if delete_controls:
        sim.remove_all_simple_controls()
        sim.remove_all_complex_controls()
    return sim

def get_tank_connections(topology):
    tank_connections = []
    for tank in topology.get_all_tanks():
        for link, _ in topology.get_adjacent_links(tank):
            tank_connections.append(link)
    return tank_connections

def set_pump_speeds(sim, pumps, speeds, time):
    for pump, speed in zip(pumps, speeds, strict=True):
        pump_speed_event = PumpSpeedEvent(pump_speed=speed, pump_id=pump, time=time)
        sim.add_actuator_event(pump_speed_event)
    return sim

def set_pump_speeds_from_dict(sim, speeds_dict, time):
    pumps, speeds = zip(*list(speeds_dict.items()))
    return set_pump_speeds(sim, pumps, speeds, time)

def read_demand_patterns(inp_file):
    sim = MyScenarioSimulator(f_inp_in=inp_file)
    pattern_idx_dict = sim.epanet_api.getNodeDemandPatternIndex()
    pattern_idxs = np.unique(np.array(list(pattern_idx_dict.values())).flatten())
    # The authors of EPyT used indexing from 1 and 0 to indicate no pattern
    pattern_idxs = [idx - 1 for idx in pattern_idxs if idx > 0]
    patterns = sim.epanet_api.getPattern()[pattern_idxs, :]
    step = sim.get_hydraulic_time_step()
    index = pd.RangeIndex(start=0, stop=patterns.shape[1]*step, step=step)
    pattern_df = pd.DataFrame(patterns.T, index=index)
    return pattern_df

def n_sensors(sensor_config):
    res = (
        len(sensor_config.bulk_species_link_sensors)
        + len(sensor_config.bulk_species_node_sensors)
        + len(sensor_config.demand_sensors)
        + len(sensor_config.flow_sensors)
        + len(sensor_config.pressure_sensors)
        + len(sensor_config.pump_efficiency_sensors)
        + len(sensor_config.pump_energyconsumption_sensors)
        + len(sensor_config.pump_state_sensors)
        + len(sensor_config.quality_link_sensors)
        + len(sensor_config.quality_node_sensors)
        + len(sensor_config.surface_species_sensors)
        + len(sensor_config.tank_volume_sensors)
        + len(sensor_config.valve_state_sensors)
    )
    return res

def obs_columns(column_desc):
    """
    Transforms a column_desc array as returned by ScadaDataExport.describe_columns
    into actual column names.
    """
    column_names = ['_'.join([entry[0], entry[1]]) for entry in column_desc]
    return column_names

def obs_columns_from_sensor_config(sensor_config):
    pressure_cols = [
        f'pressure_{pressure_sensor}'
        for pressure_sensor in sensor_config.pressure_sensors
    ]
    flow_cols = [
        f'flow_{flow_sensor}' for flow_sensor in sensor_config.flow_sensors
    ]
    demand_cols = [
        f'demand_{demand_sensor}'
        for demand_sensor in sensor_config.demand_sensors
    ]
    node_quality_cols = [
        f'node_quality_{node_quality_sensor}'
        for node_quality_sensor in sensor_config.quality_node_sensors
    ]
    link_quality_cols = [
        f'link_quality_{link_quality_sensor}'
        for link_quality_sensor in sensor_config.quality_link_sensors
    ]
    pump_state_cols = [
        f'pump_state_{pump_state_sensor}'
        for pump_state_sensor in sensor_config.pump_state_sensors
    ]
    pump_efficiency_cols = [
        f'pump_efficiency_{pump_efficiency_sensor}'
        for pump_efficiency_sensor in sensor_config.pump_efficiency_sensors
    ]
    pump_energyconsumption_cols = [
        f'pump_energyconsumption_{pump_energyconsumption_sensor}'
        for pump_energyconsumption_sensor
        in sensor_config.pump_energyconsumption_sensors
    ]
    valve_state_cols = [
        f'valve_state_{valve_state_sensor}'
        for valve_state_sensor in sensor_config.valve_state_sensors
    ]
    tank_volume_cols = [
        f'tank_volume_{tank_volume_sensor}'
        for tank_volume_sensor in sensor_config.tank_volume_sensors
    ]
    bulk_species_node_concentration_cols = [
        f'bulk_species_node_concentration_{bulk_species_node_concentration_sensor}'
        for bulk_species_node_concentration_sensor
        in sensor_config.bulk_species_node_sensors
    ]
    bulk_species_link_concentration_cols = [
        f'bulk_species_link_concentration_{bulk_species_link_concentration_sensor}'
        for bulk_species_link_concentration_sensor
        in sensor_config.bulk_species_link_sensors
    ]
    surface_species_concentration_cols = [
        f'surface_species_concentration_{surface_species_concentration_sensor}'
        for surface_species_concentration_sensor
        in sensor_config.surface_species_sensors
    ]
    all_cols = (
        pressure_cols
        + flow_cols
        + demand_cols
        + node_quality_cols
        + link_quality_cols
        + pump_state_cols
        + pump_efficiency_cols
        + pump_energyconsumption_cols
        + valve_state_cols
        + tank_volume_cols
        + bulk_species_node_concentration_cols
        + bulk_species_link_concentration_cols
        + surface_species_concentration_cols
    )
    return all_cols

def set_tank_levels2min(sim):
    api = sim.epanet_api
    min_tank_levels = api.getNodeTankMinimumWaterLevel()
    api.setNodeTankInitialLevel(min_tank_levels)
    return sim


def scenario_generator_factory(scenario_files):
    '''
    Return a generator yielding (scenario_config, scenario_simulator)-tuples for
    a collection of .inp-files.
    '''
    scenario_files = iter(scenario_files)
    def scenario_generator():
        for scenario_file in scenario_files:
            try:
                sim = MyScenarioSimulator(f_inp_in=scenario_file)
                sc = sim.get_scenario_config()
                yield sc, sim
            except StopIteration:
                print('Generated all possible scenarios')
                raise StopIteration
    return scenario_generator()

def save_lower_baseline(results_dir, rewards, objective_calculator, show=False):
    objective_calculator.to_yaml(
        path.join(results_dir, 'lower_baseline_calculator.yaml')
    )
    rewards.to_csv(path.join(results_dir, 'rewards_lower_baseline.csv'))
    fig, ax = plt.subplots()
    rewards.plot(ax=ax)
    plt.savefig(path.join(results_dir, 'lower_baseline.png'))
    if show:
        plt.show()

def constraint_based_normalization(scada, network_constraints):
    """
    Normalize SCADA data based on network constraints.

    Pressures are scaled linearly such that 0 and 1 correspond to the minimum
    and maximum pressure limit, respectively. Note that this does not
    necessarily mean that they lie in this range in case of undesired pressure
    values.
    Flows are divided by 100, yielding hektoliters per second, in order to lie
    approximately in the range [0, 1]
    Pump efficiency values are scaled linearly such that 1 corresponds to
    maximum pump efficiency.
    """
    obs_df = scada.to_pandas_dataframe()
    pressure_cols = [c for c in obs_df.columns if 'pressure' in c]
    obs_df.loc[:, pressure_cols] -= network_constraints.min_pressure
    obs_df.loc[:, pressure_cols] /= (
        network_constraints.max_pressure - network_constraints.min_pressure
    )
    flow_cols = [c for c in obs_df.columns if 'flow' in c]
    obs_df.loc[:, flow_cols] /= 100
    pump2col = lambda pump: f'pump_efficiency [] at {pump}'
    for pump, max_efficiency in network_constraints.max_pump_efficiencies.items():
        pump_col = pump2col(pump)
        if pump_col in obs_df.columns:
            obs_df.loc[:, pump_col] /= max_efficiency
        else:
            print(f'efficiency values for {pump} not part of observations')
    return obs_df.to_numpy()

def max_pump_prices(sim, max_pump_speed):
    '''
    Empirically determine the maximum pump prices throughout an episode.

    Note: So far, this actually computes the maximum energy consumption. This will
    be adjusted once a pump_price_sensor is implemented in EPyT-Flow.

    Looks very similar to max_abs_flow but might be changed in the future.
    That's why the methods are not combined.
    '''
    pumps = sim.get_topology().get_all_pumps()
    max_speeds = {pump: float(max_pump_speed) for pump in pumps}
    sim = set_pump_speeds_from_dict(sim, max_speeds, 0)
    scada = sim.run_hydraulic_simulation()
    scada_df = scada.to_pandas_dataframe()
    price_columns = [c for c in scada_df.columns if 'energyconsumption' in c]
    prices = scada_df.loc[:, price_columns]
    # Get only the pump names
    prices.rename(columns=lambda c: c.split(' ')[-1], inplace=True)
    return prices.max(axis=0)

def max_abs_flows(sim, max_pump_speed):
    '''
    Empirically determine the maximum absolute flow throughout an episode.
    '''
    pumps = sim.get_topology().get_all_pumps()
    max_speeds = {pump: float(max_pump_speed) for pump in pumps}
    sim = set_pump_speeds_from_dict(sim, max_speeds, 0)
    scada = sim.run_hydraulic_simulation()
    scada_df = scada.to_pandas_dataframe()
    flow_columns = [c for c in scada_df.columns if 'flow' in c]
    flows = scada_df.loc[:, flow_columns]
    flows.rename(columns=lambda c: c.split(' ')[-1], inplace=True)
    return flows.abs().max(axis=0)

def get_sc_and_close(sim):
    scenario_config = sim.get_scenario_config()
    sim.close()
    return scenario_config

