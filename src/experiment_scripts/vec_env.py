import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from stable_baselines3 import SAC
from gymnasium.wrappers import NormalizeObservation
from pump_control_envs import net1_env

names = lambda obj: [n for n in dir(obj) if n[0]!='_']
mshow = lambda query, obj: [n for n in dir(obj) if query in n]

env = net1_env(continuous=True, max_speed_change=0.2, max_pump_speed=1)
env = NormalizeObservation(env)
model = SAC('MlpPolicy', env, verbose=1)
