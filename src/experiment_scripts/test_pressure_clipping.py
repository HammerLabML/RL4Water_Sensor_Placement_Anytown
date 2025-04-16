import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from learn_pump_scheduler import net1_single_env 
import numpy as np

env = net1_single_env()
obs, info = env.reset()
for i in range(20):
    obs, reward, truncated, terminated, info = env.step(1)
    if (obs==-1).any():
        break
print(obs)
