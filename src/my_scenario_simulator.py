from epyt_flow.simulation import ScenarioSimulator
from epyt_flow.simulation.events.actuator_events import ActuatorEvent

class MyScenarioSimulator(ScenarioSimulator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def remove_actuator_events(self):
        self._system_events = list(
            filter(
                lambda x: not isinstance(x, ActuatorEvent),
                self._system_events
            )
        )

