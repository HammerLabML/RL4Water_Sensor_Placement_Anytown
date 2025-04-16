import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

import pandas as pd

from pump_control_envs import net1_env
from normalize_hydraulics import NormalizeHydraulics

objective_weights_file = (
    '../Data/Parameters_for_Optimization/objective_weights_4.yaml'
)
env = net1_env(objective_weights_file=objective_weights_file)
env = NormalizeHydraulics(env)
obs, info = env.reset()
print(pd.Series(dict(zip(env.obs_columns, obs))))
env.close()
