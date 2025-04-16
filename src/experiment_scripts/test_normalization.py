import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from pump_control_envs import net1_env
from normalize_hydraulics import NormalizeHydraulics
import pandas as pd

env = net1_env()
obs, info = env.reset()
pd_obs = pd.Series(obs, index=env.unwrapped.observation_desc)
env.close()

env = net1_env()
env = NormalizeHydraulics(env)
norm_obs, info = env.reset()
rec_obs = env.denormalize(norm_obs)
pd_rec = pd.Series(rec_obs, index=env.unwrapped.observation_desc)
env.close()

print((pd_obs-pd_rec).abs())

