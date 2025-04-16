import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from epyt_flow.simulation import ScenarioSimulator
from epyt_flow.simulation.events.actuator_events import PumpSpeedEvent
from epyt_flow.data.networks import load_net1

names = lambda obj: [n for n in dir(obj) if n[0]!='_']
mshow = lambda obj, name: [n for n in names(obj) if name in n]

sc = load_net1(flow_units_id=5)
sim = ScenarioSimulator(scenario_config=sc)
sim.epanet_api.deleteControls()
pump = '9'
event1 = PumpSpeedEvent(pump_speed=2.0, pump_id=pump, time=0)
event2 = PumpSpeedEvent(pump_speed=4.0, pump_id=pump, time=0)
event3 = PumpSpeedEvent(pump_speed=2.0, pump_id=pump, time=3600)
sim.add_actuator_event(event1)
