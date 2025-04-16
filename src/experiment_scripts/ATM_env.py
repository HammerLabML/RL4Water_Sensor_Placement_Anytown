import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

import numpy as np
from pump_control_envs import ATM_env

env = ATM_env(continuous=True, max_pump_speed=1)
obs1, info = env.reset()
obs2, info = env.reset()
print(obs1-obs2)

