import numpy as np

from gymnasium import ObservationWrapper, spaces
from sklearn.cluster import SpectralClustering

from pump_control_envs import ContinuousPumpControlEnv
from pump_control_envs import net3_env

import networkx as nx

from pump_control_envs import ATM_env


class SensorSelectionWrapper(ObservationWrapper):
    def __init__(self, env, sensor_file=None, obs_filter=None, seed=0):
        super(SensorSelectionWrapper, self).__init__(env)
        if not isinstance(env.unwrapped, ContinuousPumpControlEnv):
            raise AttributeError(
                'You must pass a ContinuousPumpControlEnv to this wrapper'
            )

        old_desc= env.get_wrapper_attr('observation_desc')

        if sensor_file is not None:
            with open(sensor_file, 'rb') as f:
                press_mask = np.load(f)
            if len(press_mask.shape) == 2:
                press_mask = press_mask[seed]
            self.mask = np.append(press_mask, np.ones(self.observation_space.shape[0]-press_mask.shape[0])).astype(np.bool)
        else:
            self.mask = np.ones(self.observation_space.shape[0]).astype(np.bool)

        for filter in obs_filter:
            self.mask = self.mask & np.bitwise_not(np.strings.startswith(old_desc, filter))

        shape = list(self.observation_space.shape)
        shape[-1] = self.mask.sum()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=tuple(shape), dtype=np.float32)
        self.observation_desc = np.array(old_desc)[self.mask]

    def observation(self, observation):
        if len(self.observation_space.shape) == 2:
            return observation[:,self.mask]
        else:
            return observation[self.mask]


class SensorPlacementGenerator():
    def __init__(self, env, seed = 0):
        self.n_pressure_nodes = np.strings.count(env.get_wrapper_attr('observation_desc'), 'press').sum()
        np.random.seed(seed)

    def create_mask(self, n_sensors : int = 8, selection_method :  str ="random"):
        assert n_sensors > 0 and n_sensors <= self.n_pressure_nodes

        if selection_method == "random":
            mask = np.zeros(self.n_pressure_nodes)
            mask[:n_sensors] = 1
            np.random.shuffle(mask)
            return mask
        else:
            raise NotImplementedError

if __name__=='__main__':

    methods = ['random']
    n_sensors = [1, 2, 4, 8]
    repetitions = range(5) # relevant for placements with random component
    env_name = 'ATM_env'
    path = '../Data/' + 'Anytown' + '/'

    if env_name == 'Net3':
        env = net3_env()
    elif env_name == 'ATM_env':
        env = ATM_env()

    gen = SensorPlacementGenerator(env, seed = 0)


    for method in methods:
        for n in n_sensors:
                if method == 'random':
                    mask = np.array([gen.create_mask(n, method) for r in repetitions])
                else:
                    mask = gen.create_mask(n, method)
                with open(path + 'sensor_placement_' + method + '_' + str(n) + '_sensors.npy', 'wb') as f:
                    np.save(f, mask)

    env.close()

