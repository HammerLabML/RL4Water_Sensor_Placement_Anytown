from epyt_flow.gym import ScenarioControlEnv
from epyt_flow.simulation.events.actuator_events import ActuatorConstants
from epyt_flow.simulation import ScenarioSimulator

class PumpStateControlEnv(ScenarioControlEnv):
    '''This env turns pumps on or off.'''

    def __init__(self, scenario_config, **kwargs):
        super().__init__(scenario_config, **kwargs)
        with ScenarioSimulator(scenario_config=scenario_config) as dummy_sim:
            self.pumps = dummy_sim.get_topology().get_all_pumps()

    def step(self, actions):
        for (pump, action) in zip(self.pumps, actions, strict=True):
            if action:
                self.set_pump_status(pump, ActuatorConstants.EN_OPEN)
            else:
                self.set_pump_status(pump, ActuatorConstants.EN_CLOSED)
        reward = 0
        if self.autoreset:
            scada = self._next_sim_itr()
            return scada, reward
        else:
            scada, terminated = self._next_sim_itr()
            return scada, reward, terminated

if __name__=='__main__':
    with ScenarioSimulator(f_inp_in='net1.inp') as dummy_sim:
        dummy_sim.place_pump_state_sensors_everywhere()
        scenario_config = dummy_sim.get_scenario_config()
    env = PumpStateControlEnv(scenario_config, autoreset=False)
    scada, terminated = env.reset()
    print(f'Initial pump status: {scada.get_data_pumps_state()[0,0]}')
    print('Opening pump...')
    scada, _, terminated = env.step([1])
    print(f'Current pump state: {scada.get_data_pumps_state()[0,0]}')
    print('Closing pump...')
    scada, _, terminated = env.step([0])
    print(f'Current pump state: {scada.get_data_pumps_state()[0,0]}')
    while not terminated:
        scada, _, terminated = env.step([1])
    env.close()

