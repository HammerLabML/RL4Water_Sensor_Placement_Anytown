from gymnasium.wrappers import NormalizeObservation
import numpy as np

class MyNormalizeObservation(NormalizeObservation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_statistics = True

    def stop_updating_statistics(self):
        self._update_statistics = False

    def normalize(self, obs):
        """
        Update the statistics, unless stop_updating_statistics has been called in
        the past.
        Normalises the observation using the running mean and variance computed from
        previous observations.
        """
        if self._update_statistics:
            self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)

