from epyt_flow.simulation import ScenarioSimulator
from epyt_flow.simulation.events.actuator_events import PumpSpeedEvent

import matplotlib.pyplot as plt

def compare_pump_speeds(inp_file, speed_dicts):
    sim = ScenarioSimulator(inp_file)
    sim.epanet_api.deleteControls()
    sim.set_general_parameters(simulation_duration=60)
    sim.place_pressure_sensors_everywhere()
    sim.place_pump_efficiency_sensors_everywhere()
    sim.place_pump_state_sensors_everywhere()
    res = []
    for speed_dict in speed_dicts:
        for pump, speed in speed_dict.items():
            pump_speed_event = PumpSpeedEvent(
                pump_speed=speed, pump_id=pump, time=0
            )
            sim.add_actuator_event(pump_speed_event)
        scada = sim.run_hydraulic_simulation()
        scada_df = scada.to_pandas_dataframe()
        # Retrieve values at time 0
        res.append(scada_df.loc[0, :])
        # Remove all events before the next trial
        sim._system_events = list(
            filter(lambda ev: not isinstance(ev, PumpSpeedEvent), sim._system_events)
        )
    sim.close()
    return res

if __name__=='__main__':
    inp_file = 'ATM.inp'
    # Note: Only the order is different here
    speed_dict1 = {'b1': 1.0, 'b2': 2.0, 'b3': 3.0}
    speed_dict2 = {'b1': 1.0, 'b3': 3.0, 'b2': 2.0}
    res1, res2 = compare_pump_speeds(inp_file, [speed_dict1, speed_dict2])
    efficiency_idxs = [idx for idx in res1.index if 'efficiency' in idx]
    print(res1[efficiency_idxs])
    print(res2[efficiency_idxs])

