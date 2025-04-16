import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from learn_pump_scheduler import net1_single_env 
from stable_baselines3 import DQN
import numpy as np
from stable_baselines3.common.vec_env.util import obs_space_info

env = net1_single_env()
obs, info = env.reset()
print("Normal Env: Pump Speed after first reset")
print(obs.flatten()[12])
print('Taking a couple of steps...')
for i in range(10):
    env.step(2)
obs, info = env.reset()
print("Normal Env: Pump Speed after second reset")
print(obs.flatten()[12])

model = DQN('MlpPolicy', env)
model.set_parameters('../Results/Single_Network/Net1/model_1')
vec_env = model.get_env()
obs = vec_env.reset()
print('Vec Env: Pump Speed after first reset')
print(obs.flatten()[12])
print('Taking a couple of steps...')
for i in range(10):
    vec_env.step(np.array([2]))
obs = vec_env.reset()
print('Vec Env: Pump Speed after second reset')
print(obs.flatten()[12])
print('One step later...')
obs, _, _, _ = vec_env.step(np.array([2]))
print(obs.flatten()[12])
