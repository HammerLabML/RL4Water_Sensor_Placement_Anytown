import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from pump_control_envs import net1_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from time_encoders import EpisodeStepsEncoder

names = lambda obj: [n for n in dir(obj) if n[0]!='_']
mshow = lambda obj, m: [n for n in dir(obj) if m in n]
if __name__=='__main__':
    env = make_vec_env(
        env_id=net1_env,
        n_envs=2,
        wrapper_class=EpisodeStepsEncoder,
        vec_env_cls=DummyVecEnv
    )
#    env = net1_env()
    print(env.env_is_wrapped(EpisodeStepsEncoder))
    env.close()

