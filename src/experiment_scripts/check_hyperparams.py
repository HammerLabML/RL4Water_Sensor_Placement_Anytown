import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from stable_baselines3 import DQN
from pump_control_envs import anytown_env

names = lambda obj: [name for name in dir(obj) if name[0]!='_']
mshow = lambda obj, n: [name for name in names(obj) if n in name]

env = anytown_env()
model_path = '../Results/No_Uncertainty/Anytown/model_4.zip'
agent = DQN('MlpPolicy', env)
agent.set_parameters(model_path)
