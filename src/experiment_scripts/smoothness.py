import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from pump_control_envs import ContinuousPumpControlEnv
from speed_aware_observation import SpeedAwareObservation
import numpy as np
from os import path

data_dir = '../Data/Net1'
network_file = path.join(data_dir, 'net1.inp')
initial_pump_speeds_file = path.join(data_dir, 'initial_pump_speeds.yaml')
network_constraints_file = path.join(data_dir, 'net1_constraints.yaml')
objective_weights_file = '../Data/Parameters_for_Optimization/objective_weights_3.yaml'
env = ContinuousPumpControlEnv.from_files(
    network_file, initial_pump_speeds_file, network_constraints_file,
    objective_weights_file, max_pump_speed=1
)
env = SpeedAwareObservation(env, n_timesteps=3)
obs, info = env.reset()
prep = lambda x: np.array([x], dtype=np.float32)
for speed in [.5, .7, .5]:
    obs, rew, term, trunc, info = env.step(prep(speed))
print(info['reward_info'])
env.close()

