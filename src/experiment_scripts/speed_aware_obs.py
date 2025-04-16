import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from pump_control_envs import net1_env, anytown_env
from speed_aware_observation import SpeedAwareObservation
import numpy as np

env = net1_env(continuous=True)
env = SpeedAwareObservation(env, n_timesteps=3)
obs, info = env.reset()
prep = lambda x: np.array([x], dtype=np.float32)
speeds = [.5, .4, .8, .9]
for speed in speeds:
    obs, rew, term, trunc, info = env.step(prep(speed))
print(obs)
env.close()
