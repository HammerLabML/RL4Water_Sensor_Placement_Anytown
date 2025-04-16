import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from os import path
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common import logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from pump_control_envs import net1_env

def log_cartpole():
    env = gym.make('CartPole-v1')
    new_logger = logger.configure('cartpole_logs', ['stdout', 'csv'])
    model = DQN('MlpPolicy', env)
    model.set_logger(new_logger)
    model.learn(total_timesteps=10_000)

def log_net1():
    env = net1_env(continuous=False)
    log_dir = 'net1_logs'
    env = Monitor(env, path.join(log_dir, 'monitor.csv'))
    env = DummyVecEnv([lambda: env])
    new_logger = logger.configure(log_dir, ['stdout', 'csv'])
    model = DQN('MlpPolicy', env, verbose=1)
    model.set_logger(new_logger)
    model.learn(10_000)

log_net1()
