import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
from gymnasium.wrappers import RescaleAction, TimeAwareObservation,\
NormalizeObservation, FrameStackObservation, TimeLimit
from stable_baselines3 import DQN, SAC
from stable_baselines3.common import logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv,\
VecNormalize, VecMonitor, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
import os
from os import path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import gym_util
from pump_control_envs import anytown_env, ATM_env, net1_env, leakdb_env
from save_best_model_callback import SaveBestModelCallback
from speed_aware_observation import SpeedAwareObservation
from normalize_hydraulics import NormalizeHydraulics
from time_encoders import EpisodeStepsEncoder, TimeOfDayEncoder, WeekdayEncoder

def evaluate(model, n_episodes=5, plot_changes=True, print_actions=False):
    vec_env = model.get_env()
    for episode in range(n_episodes):
        obs = vec_env.reset()
        if plot_changes:
            fig = plt.figure()
            plt.axis('off')
            img = plt.imshow(vec_env.render())
            plt.show(block=False)
            plt.pause(0.1)
        done = False
        rewards = []
        actions = []
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            actions.append(action[0])
            obs, reward, done, info = vec_env.step(action)
            rewards.append(reward[0])
            if plot_changes:
                img.set_data(vec_env.render())
                plt.show(block=False)
                plt.pause(0.1)
        if print_actions:
            actions = np.array(actions)
            unique_actions, counts = np.unique(actions, return_counts=True)
            print(f'Action counts for episode {episode}')
            print(dict(zip(unique_actions, counts)))
        else:
            print(f'Rewards for episode {episode}')
            print(np.array(rewards).mean())

def random_baseline(pump_control_env):
    obs, info = pump_control_env.reset()
    plt.figure()
    plt.axis('off')
    img = plt.imshow(pump_control_env.render())
    plt.show(block=False)
    plt.pause(0.1)
    rewards = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = pump_control_env.action_space.sample()
        obs, reward, terminated, truncated, info = pump_control_env.step(action)
        img.set_data(pump_control_env.render())
        plt.show(block=False)
        plt.pause(0.1)
        rewards.append(reward)
    print('Rewards for an episode with random actions')
    print(np.array(rewards).mean())

if __name__=='__main__':
    results_dir = '../Results/Multi_Network/LeakDB'
    model_nr = '6'
    model_path = path.join(results_dir, f'model_{model_nr}')
    log_dir = model_path + '_log'
    os.makedirs(log_dir, exist_ok=True)
    objective_weights_file = (
        '../Data/Parameters_for_Optimization/objective_weights_5.yaml'
    )
    make_speed_aware = lambda x: SpeedAwareObservation(x, n_timesteps=3)
    add_previous_obs = lambda x: FrameStackObservation(x, stack_size=2)
    add_time_limit = lambda x: TimeLimit(x, x.unwrapped._max_itr_till_truncation)
    make_time_aware = lambda x: EpisodeStepsEncoder(x)
    rescale_action = lambda x: RescaleAction(x, -1, 1)

    pump_control_env = make_vec_env(
        env_id = lambda **kwargs: leakdb_env(scenario_nrs=list(range(1, 301)), **kwargs),
        env_kwargs = dict(
            max_pump_speed=1,
            objective_weights_file=objective_weights_file
        ),
        n_envs = 9,
        seed = 42,
        monitor_dir = log_dir,
        wrapper_class = lambda x: TimeOfDayEncoder(WeekdayEncoder(NormalizeHydraulics(rescale_action(x)))),
        vec_env_cls = SubprocVecEnv
    )
#    pump_control_env = leakdb_env(
#        scenario_nrs=list(range(1, 301)),
#        max_pump_speed=1,
#        objective_weights_file=objective_weights_file
#    )
#    pump_control_env = Monitor(pump_control_env, log_dir)
#    pump_control_env = make_time_aware(NormalizeHydraulics(rescale_action(pump_control_env)))
    model = SAC(
        'MlpPolicy', pump_control_env,
        verbose=1, seed=42
    )
    new_logger = logger.configure(log_dir, ['stdout', 'csv', 'tensorboard'])
    model.set_logger(new_logger)
    callback = SaveBestModelCallback(
        check_freq=1000, log_dir=log_dir,
        save_norm_stats=False, verbose=False
    )
    model.learn(total_timesteps=150_000, callback=callback)
    model.save(model_path)
    pump_control_env.close()

