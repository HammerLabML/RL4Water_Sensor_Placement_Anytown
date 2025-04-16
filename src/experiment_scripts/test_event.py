from epyt_flow.simulation.events.actuator_events import PumpSpeedEvent
from epyt_flow.simulation import ScenarioSimulator as SIM

def net1_sim():
    sim = SIM('net1.inp')
    sim.epanet_api.deleteControls()
    sim.place_pressure_sensors_everywhere()
    return sim

speed_2_0 = PumpSpeedEvent(pump_speed=2., pump_id='9', time=0)
speed_2_300 = PumpSpeedEvent(pump_speed=2., pump_id='9', time=300)

sim1 = net1_sim()
sim1.add_actuator_event(speed_2_0)
scada1 = sim1.run_hydraulic_simulation()
df1 = scada1.to_pandas_dataframe()

sim2 = net1_sim()
sim2.add_actuator_event(speed_2_0)
sim2.add_actuator_event(speed_2_300)
scada2 = sim2.run_hydraulic_simulation()
df2 = scada2.to_pandas_dataframe()

sim3 = net1_sim()
scada3 = sim3.run_hydraulic_simulation()
df3 = scada3.to_pandas_dataframe()

print((df1-df3).head().abs().max(axis=1))


