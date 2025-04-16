import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from pump_control_envs import net1_env
from normalize_hydraulics import NormalizeHydraulics
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC
from gymnasium.wrappers import TimeAwareObservation, RescaleAction

if __name__=='__main__':
    monitor_dir = 'test_log'
#    train_env = make_vec_env(
#        env_id=net1_env,
#        n_envs=5,
#        seed=42,
#        monitor_dir=monitor_dir,
#        wrapper_class=TimeAwareObservation,
#        vec_env_cls=SubprocVecEnv
#    )
    train_env = net1_env()
    train_env = Monitor(train_env, monitor_dir)
    train_env = RescaleAction(train_env, min_action=-1, max_action=1)
    train_env = TimeAwareObservation(NormalizeHydraulics(train_env))
    train_model = SAC('MlpPolicy', train_env, verbose=1)
    train_model.learn(total_timesteps=1000)
    train_model.save('model.zip')
    train_env.close()

#    test_env = make_vec_env(
#        env_id=net1_env,
#        seed=42,
#        wrapper_class=TimeAwareObservation,
#        vec_env_cls=DummyVecEnv
#    )
    test_env = net1_env()
    test_env = RescaleAction(test_env, min_action=-1, max_action=1)
    test_env = TimeAwareObservation(NormalizeHydraulics(test_env))
    test_model = SAC('MlpPolicy', test_env, verbose=1)
    test_model.set_parameters('model.zip')
    test_env.close()

