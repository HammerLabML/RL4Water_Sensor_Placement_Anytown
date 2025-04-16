import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from pump_control_envs import net1_env, anytown_env
from gym_util import to_multidiscrete
import pandas as pd

names = lambda obj: [n for n in dir(obj) if n[0]!='_']
mshow = lambda obj, m: [n for n in names(obj) if m in n]

def to_df(obs, info):
    columns = ['_'.join([row[0], row[1]]) for row in info['data_desc']]
    df = pd.DataFrame(obs.reshape(-1, len(columns)), columns=columns)
    df.rename(columns={'time_': 'time'}, inplace=True)
    df.index = df.time
    df.drop(columns='time', inplace=True)
    return df

def only_pressures(df):
    pressure_columns = [c for c in df.columns if 'pressure' in c]
    return df.loc[:, pressure_columns]

def only_flows(df):
    flow_columns = [c for c in df.columns if 'flow' in c]
    return df.loc[:, flow_columns]

def only_speeds(df):
    speed_columns = [c for c in df.columns if 'speed' in c]
    return df.loc[:, speed_columns]

env = anytown_env(continuous=False)
mas = env._multidiscrete_action_space
for action in range(27):
    print(f'{action}: {to_multidiscrete(action, mas)}')
exit()
old_obs, info = env.reset()
old_obs_df = to_df(old_obs1, info)
same_speed_obs = env.step(0)[0]
same_speed_df = to_df(same_speed_obs, info)
env.reset()
high_speed_obs = env.step(26)[0]
high_speed_df = to_df(high_speed_obs, info)
env.close()

