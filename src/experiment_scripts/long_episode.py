import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

import sim_util
from pump_control_envs import leakdb_env

env = leakdb_env(scenario_nrs=[1,2,3,4], continuous=True, max_pump_speed=1, max_itr_till_truncation=337)
n_steps = 0
terminated, truncated = False, False
obs, info = env.reset()
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    n_steps += 1
print(info['sim_time'])

