import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from pump_control_envs import net1_env
import sim_util

env = net1_env()
obs, info = env.reset()
sensor_config = env._scenario_config.sensor_config
oc1 = sim_util.obs_columns(info['data_desc'])
oc2 = sim_util.obs_columns_from_sensor_config(sensor_config)
if len(oc1)==len(oc2):
    if all([oc1[i]==oc2[i] for i in range(len(oc2))]):
        print('OK')
    else:
        print('Content Missmatch')
else:
    print('Length Missmatch')
env.close()

