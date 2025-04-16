import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

import sim_util
from my_scenario_simulator import MyScenarioSimulator
from epyt_flow.simulation.events.leakages import AbruptLeakage

names = lambda obj: [n for n in dir(obj) if n[0]!='_']
mshow = lambda obj, m: [n for n in dir(obj) if m in n]

def dummy_sim():
    inp_file = '../Data/Net1/net1.inp'
    sim = MyScenarioSimulator(f_inp_in=inp_file)
    sim.set_general_parameters(flow_units_id=8) # cubic meters per hour
    topo = sim.get_topology()
    sim.set_pressure_sensors(topo.get_all_junctions())
    sim.place_pump_sensors_everywhere()
    return sim

def add_dummy_speed_event(sim):
    topo = sim.get_topology()
    pump = topo.get_all_pumps()[0]
    speed = 1.0
    time = 0
    sim = sim_util.set_pump_speeds(sim, [pump], [speed], time)
    return sim

def add_dummy_leak(sim):
    link_with_leak = '22'
    leak = AbruptLeakage(
        link_id=link_with_leak,
        diameter=0.1,
        start_time=0,
        end_time=10800
    )
    sim.add_leakage(leak)
    return sim

def test_removal():
    sim = dummy_sim()
    sim = add_dummy_speed_event(sim)
    sim = add_dummy_leak(sim)
    print('System events before actuator events removal')
    print(sim.system_events)
    sim.remove_actuator_events()
    print('System events after actuator event removal')
    print(sim.system_events)
    sim.close()

def test_side_effects():
    sim1 = dummy_sim()
    sim1 = add_dummy_speed_event(sim1)
    sim1.remove_actuator_events()
    scada1 = sim1.run_hydraulic_simulation()
    msm1 = scada1.get_data()
    sim1.close()
    sim2 = dummy_sim()
    scada2 = sim2.run_hydraulic_simulation()
    msm2 = scada2.get_data()
    print('Testing if measurements are equal...')
    print((msm1==msm2).all(axis=None))

def test_reconstruction_from_sensor_config(remove_event=True):
    sim = dummy_sim()
    sim = add_dummy_speed_event(sim)
    print('Current actuator events')
    print(sim.actuator_events)
    if remove_event:
        sim.remove_actuator_events()
        print('After removal')
        print(sim.actuator_events)
    new_sim = MyScenarioSimulator(scenario_config=sim.get_scenario_config())
    print('After reconstruction')
    print(new_sim.actuator_events)

if __name__=='__main__':
    test_reconstruction_from_sensor_config(remove_event=False)

