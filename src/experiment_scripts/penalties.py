import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

import numpy as np
import pandas as pd

from pump_control_envs import net1_env

env = net1_env(objective_weights_file='experiment_scripts/example_penalties.yaml')
to_act = lambda x: np.array([x], dtype=np.float32)
obs, info = env.reset()
output = pd.DataFrame(columns=['action', 'pressure_obj', 'smoothness_obj'])
actions = [.6, .6, .6, .3, .3, .3]
for i, action in enumerate(actions):
    obs, reward, terminated, truncated, info = env.step(to_act(action))
    output.loc[i] = dict(
        action=action,
        pressure_obj=info['reward_info']['pressure_obj'],
        smoothness_obj=info['reward_info']['smoothness_obj']
    )
print(output)
print(
    f'Used objective calculator:\n {env.objective_calculator}'
)
env.close()

