import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from pump_control_envs import net1_env, leakdb_env

tmpdir = os.environ['TMPDIR']
print(f'Initial number of tmpfiles: {len(os.listdir(tmpdir))}')
env = leakdb_env(scenario_nrs=list(range(1, 20)))
for i in range(10):
    env.reset()
    print(f'Number of tmpfiles: {len(os.listdir(tmpdir))}')
env.close()
print(f'Final number of tmpfiles: {len(os.listdir(tmpdir))}')

