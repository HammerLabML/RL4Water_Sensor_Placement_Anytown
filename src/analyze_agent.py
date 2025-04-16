from pump_control_envs import PumpControlEnv, anytown_env, net1_env, ATM_env, leakdb_env
from gymnasium import Env
import pandas as pd
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv,\
VecNormalize, VecMonitor, VecFrameStack

import matplotlib.pyplot as plt
import numpy as np
from os import path
from gymnasium.wrappers import TimeAwareObservation, NormalizeObservation, RescaleAction, FrameStackObservation
from torch.utils.tensorboard import SummaryWriter

import gym_util
import sim_util
import sb3_util
from speed_aware_observation import SpeedAwareObservation
from dummy_agents import ConstantEqualAgent
from normalize_hydraulics import NormalizeHydraulics
from time_encoders import EpisodeStepsEncoder, WeekdayEncoder, TimeOfDayEncoder

class AgentAnalyzer():
    """
    Analyze an agent in a PumpControlEnv.

    This class can be used to produce plots and prints for analysis and
    debugging of Reinforcement Learning agents.

    @param env: An environment based on a PumpControlEnv, i.e. one of the following:
    1. a basic PumpControlEnv
    2. a PumpControlEnv wrapped with a gymnasium wrapper
    3. an instance of stable_baselines3.common.vec_env.VecEnv comprised of
      sub-environments of type 1 or 2.
    4. A VecEnv (type 3) wrapped in a stable_baselines3.common.vec_env.VecEnvWrapper

    @param agent: stable_baselines3.common.base_class.BaseAlgorithm
    the agent performing actions in the environment
    """

    def __init__(self, env, agent):
        self.env = env
        try:
            # call the setter method for self._agent and see if it goes through
            self.agent = agent
        except ValueError as err:
            print("Observation and/or action space of agent don't match env")
            raise err

    def _check_spaces(self, observation_space, action_space):
        if not observation_space==self._env.observation_space:
            raise ValueError(
                f'observation spaces must be the same. '
                f'need {self._env.observation_space} '
                f'but got {observation_space}'
            )
        if not action_space==self._env.action_space:
            raise ValueError(
                f'action spaces must be the same. '
                f'need {self._env.action_space} '
                f'but got {action_space}'
            )

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, env):
        pce = True # Assume that env is a PumpControlEnv (or composed of those)
        if isinstance(env, Env):
            if not isinstance(env.unwrapped, PumpControlEnv):
                pce = False
        elif isinstance(env, VecEnv):
            if env.num_envs > 1:
                raise ValueError(
                    f'Vectorized environments containing more than one '
                    f'sub-environment are not supported in AgentAnalyzer.'
                )
            sub_env = env.get_attr('unwrapped')[0]
            if not isinstance(sub_env, PumpControlEnv):
                pce = False
        else: # Neither Env nore VecEnv
            raise AttributeError(
                f'env must be either a gymnasium.Env of a '
                f'stable_baselines3.common.vec_env.VecEnv. '
                f'You passed an instance of {type(env)}'
            )
        if not pce:
            raise ValueError('env must be a PumpControlEnv or composed of those.')
        if hasattr(self, '_env'): # will be True after __init__
            self._check_spaces(env.observation_space, env.action_space)
        self._env = env

    @property
    def agent(self):
        return self._agent

    @agent.setter
    def agent(self, agent):
        vec_env = agent.get_env() # The env the agent was trained on
        self._check_spaces(vec_env.observation_space, vec_env.action_space)
        self._agent = agent

    def record_env_obs(self, obs, denormalize=True):
        """
        Reconstruct the original observation of a wrapped environment.

        The following assumptions are made:
        - Frame stacking (if any) is applied AFTER normalization (if any)
        - The environment is normalized either with NormalizeObservation or
          with NormalizeHydraulics or none of them, but NOT with both.

        @param obs: np.ndarray,
        The observation as returned by the wrapped environment

        @param denormalize: bool, optional, default=True
        Whether normalization should be reversed. If not, only frame-stacking is
        reversed (if applicable).

        @returns the original observation (np.ndarray)
        """
        has_stacked_obs = gym_util.is_wrapped_with(self._env, FrameStackObservation)
        if has_stacked_obs:
            obs = obs[-1]
        if denormalize:
            if gym_util.is_wrapped_with(self._env, NormalizeObservation):
                obs_rms = self._env.get_wrapper_attr('obs_rms')
                mean = obs_rms.mean
                std = np.sqrt(obs_rms.var)
                obs = (obs * std) + mean
            elif gym_util.is_wrapped_with(self._env, NormalizeHydraulics):
                obs = self._env.get_wrapper_attr('denormalize')(obs)
        return obs

    def run_episode_basic_env(self,
            return_qvalues=False, denormalize_rewards=True,
            denormalize_obs=True, return_actions=False):
        '''See documentation of run_episode.'''
        if return_qvalues and not isinstance(self.agent, DQN):
            raise RuntimeError('Can only return q values for DQN agents')
        obs, info = self._env.reset()
        speeds = pd.DataFrame(columns=self._env.unwrapped.pumps)
        speeds.loc[0, :] = self._env.unwrapped.current_pump_speeds
        rewards = pd.DataFrame(columns=info['reward_info'].keys())
        rewards.loc[0, :] = info['reward_info']
        observation_desc = self.env.get_wrapper_attr('observation_desc')
        is_time_aware = gym_util.is_wrapped_with(self._env, TimeAwareObservation)
        if is_time_aware:
            observation_desc.append('episode_steps')
        episode_obs = pd.DataFrame(columns=observation_desc)
        recorded_obs = self.record_env_obs(obs, denormalize=denormalize_obs)
        episode_obs.loc[0, :] = dict(zip(observation_desc, recorded_obs))
        if return_qvalues:
            episode_qvalues = pd.DataFrame(columns=['nop', 'dec', 'inc'])
            qvalues = sb3_util.qvalues(self._agent, obs)
            episode_qvalues.loc[0, :] = dict(
                zip(episode_qvalues.columns, qvalues)
            )
        terminated = False
        truncated = False
        all_actions = []
        while not (terminated or truncated):
            action, _ = self._agent.predict(obs, deterministic=True)
            all_actions.append(action)
            obs, reward, terminated, truncated, info = self._env.step(action)
            time = info['sim_time']
            speeds.loc[time, :] = self._env.unwrapped.current_pump_speeds
            rewards.loc[time, :] = info['reward_info']
            recorded_obs = self.record_env_obs(obs, denormalize=denormalize_obs)
            episode_obs.loc[time, :] = dict(zip(observation_desc, recorded_obs))
            if return_qvalues:
                qvalues = sb3_util.qvalues(self._agent, obs)
                episode_qvalues.loc[time, :] = dict(
                    zip(episode_qvalues.columns, qvalues)
                )
        if denormalize_rewards:
            # Rewards are assumed to be centered around 0
            # and are now denormalized to lie in [0, 1] instead
            rewards['reward'] += 0.5
        pandas_objects = [speeds, rewards, episode_obs]
        if return_qvalues:
            pandas_objects.append(episode_qvalues)
        for pandas_object in pandas_objects:
            pandas_object.index.name = 'Time'
        if return_actions:
            all_actions = np.array(all_actions)
            return tuple(pandas_objects + [all_actions])
        return tuple(pandas_objects)

    def record_venv_obs(self, obs_df, obs, time, denormalize_obs=True):
        if gym_util.is_wrapped_with(self._env, VecNormalize) and denormalize_obs:
            obs = self._env.get_original_obs()[0]
        if gym_util.is_wrapped_with(self._env, VecFrameStack):
            obs_shape = obs_df.shape[1]
            obs = obs[-obs_shape:]
        if self.env.env_is_wrapped(FrameStackObservation)[0]:
            obs = obs[1]
        if self.env.env_is_wrapped(NormalizeHydraulics)[0]:
            observation_desc = self.env.get_attr('get_wrapper_attr')[0](
                'observation_desc'
            )
            obs = self.env.get_attr('denormalize')[0](obs, observation_desc)
        obs_df.loc[time, :] = obs

    def run_episode_vec_env(self,
            return_qvalues=False, denormalize_rewards=True,
            denormalize_obs=True, return_actions=False):
        '''See documentation of run_episode.'''
        if return_qvalues and not isinstance(self.agent, DQN):
            raise RuntimeError('Can only return q values for DQN agents')
        obss = self._env.reset()
        # This function retrieves attributes from wrappers around
        # the first inner sub-environment
        get_wrapper_attr = self._env.get_attr('get_wrapper_attr')[0]
        # We need this to generate column names for the observations
        observation_desc = get_wrapper_attr('observation_desc')
        episode_obs = pd.DataFrame(columns=observation_desc)
        self.record_venv_obs(
            episode_obs, obss[0], 0,
            denormalize_obs=denormalize_obs
        )
        unwrapped = self._env.unwrapped
        episode_speeds = pd.DataFrame(columns=unwrapped.get_attr('pumps')[0])
        episode_speeds.loc[0, :] = unwrapped.get_attr('current_pump_speeds')[0]
        # Can be initialized only from info after step 1
        episode_rewards = None
        if return_qvalues:
            # TODO: Test if this still works
            episode_qvalues = pd.DataFrame(columns=['nop', 'dec', 'inc'])
            qvalues = sb3_util.qvalues(self._agent, obss)
            episode_qvalues.loc[0, :] = dict(
                zip(episode_qvalues.columns, qvalues)
            )
        done = False
        all_actions = []
        while not done:
            actions, _ = self._agent.predict(obss, deterministic=True)
            all_actions.append(actions[0])
            obss, rewards, dones, infos = self._env.step(actions)
            reward, done, info = map(lambda l: l[0], [rewards, dones, infos])
            if episode_rewards is None:
                episode_rewards = pd.DataFrame(columns=info['reward_info'].keys())
            time = info['sim_time']
            episode_speeds.loc[time, :] = (
                self._env.unwrapped.get_attr('current_pump_speeds')[0]
            )
            episode_rewards.loc[time, :] = info['reward_info']
            self.record_venv_obs(
                episode_obs, obss[0], time, denormalize_obs=denormalize_obs
            )
            if return_qvalues:
                qvalues = sb3_util.qvalues(self._agent, obss)
                episode_qvalues.loc[time, :] = dict(
                    zip(episode_qvalues.columns, qvalues)
                )
        if denormalize_rewards:
            # Rewards are assumed to be centered around 0
            # and are now denormalized to lie in [0, 1] instead
            episode_rewards['reward'] += 0.5
        pandas_objects = [episode_speeds, episode_rewards, episode_obs]
        if return_qvalues:
            pandas_objects.append(episode_qvalues)
        for pandas_object in pandas_objects:
            pandas_object.index.name = 'Time'
        if return_actions:
            all_actions = np.array(all_actions)
            return tuple(pandas_objects + [all_actions])
        return tuple(pandas_objects)

    def run_episode(self, **kwargs):
        '''
        Run an episode with self.agent acting on self.env

        @param return_qvalues, bool, optional, default=False
        applicable only for DQN policies. In addition to other returns, return a
        pandas DataFrame containing Q-values for each action at each timestep.

        @param denormalize_rewards, bool, optional, default=True
        Add 0.5 to the rewards (assuming 0.5 was subtracted to center them at 0
        during training)

        @param denormalize_obs, bool, optional, default=True applicable if
        self.env was wrapped with NormalizeObservation (in case of
        gymnasium.Env) or with VecNormalize (in case of
        stable_baselines3.common.vec_env.VecEnv). If True, denormalize the
        observations before returning them.
        Note: Denormalization for the NormalizeHydraulics wrapper is not
        implemented here, yet.

        @param return_actions, bool, default=False
        If True, return the actions taken at each timestep along with other
        output.

        @returns
        - speeds (pandas.DataFrame) speeds of pumps (columns) at each time
        - rewards (pandas.DataFrame) all reward components at each time
            column names are: ['pressure_obj', 'tank_obj',
            'pump_efficiency_obj', 'smoothness_obj', 'pump_price_obj',
            'reward', 'centered']
            'reward' contains the weighted result. 'centered' is a bool
            indicating whether the reward was centered around 0 during training
        - episode_obs (pandas.DataFrame) observations at each time step. Column
          names take the form '<measurement>_<location>' (e.g. 'pressure_17')
        - episode_qvalues (pandas.DataFrame), only returned if return_qvalues=True,
          Q-values for each action (columns) at each time
        - actions (numpy.ndarray), only returned if return_actions=True,
          actions at each time step. Usually, speeds should be sufficient.
          Actions might be useful for debugging.
        '''
        if isinstance(self._env, Env):
            return self.run_episode_basic_env(**kwargs)
        elif isinstance(self._env, VecEnv):
            return self.run_episode_vec_env(**kwargs)

    def close(self):
        self._env.close()

    def plot_speeds_and_rewards(
            self, speeds, rewards,
            rewards_lower_std=None, rewards_upper_std=None,
            rewards_global_min=None,
            output_file=None):
        '''
        Plot pump speeds and rewards.

        If lower and upper standard deviation for the reward are given, they are
        included in the plot

        @param speeds: pandas.Series or DataFrame, pump speeds
        @param rewards: pandas.Series or DataFrame, rewards for each timestep
        series should be given when lower and upper std are used to avoid messy
        plots
        @param rewards_lower_std: pd.Series, default=None
        mean of rewards - standard deviation
        @param rewards_upper_std: pd.Series, default=None,
        mean of rewards + standard deviation
        @param output file: str, default=None,
        file to save the plot to. If None, the plot is shown immediately
        '''
        fig, axs = plt.subplots(1, 2)
        speed_ax, reward_ax = axs
        speeds.plot(ax=speed_ax)
        speed_ax.set_title('Pump Speeds')
        if isinstance(rewards, pd.Series):
            if rewards_lower_std is not None or rewards_upper_std is not None:
                rewards_label = 'mean reward'
            else:
                rewards_label = None
            rewards.plot(ax=reward_ax, color='red', label=rewards_label)
            # A global minimum is the minimum over all trials and timesteps,
            # rather than the minimum of the means
            if rewards_global_min is None:
                min_reward = rewards.min()
            else:
                min_reward = rewards_global_min
            plt.suptitle(
                f'Average Reward: {rewards.mean():.4f}\n'
                f'Minimum Reward: {min_reward:.4f}'
            )
            plt.tight_layout()
        else:
            rewards.plot(ax=reward_ax)
        reward_ax.set_title('Rewards')
        if rewards_lower_std is not None:
            rewards_lower_std.plot(
                ax=reward_ax, alpha=0.5, linestyle='--',
                label='reward lower std'
            )
        if rewards_upper_std is not None:
            rewards_upper_std.plot(
                ax=reward_ax, alpha=0.5, linestyle='--',
                label='reward upper std'
            )
        for ax in axs:
            ax.set_xlabel('Time in seconds')
        plt.legend()
        if output_file is not None:
            plt.savefig(output_file)
        else:
            plt.show()

    def compare_test_episodes(self, num_trials=10, output_file=None):
        all_rewards = []
        all_obs = []
        for i in range(num_trials):
            speeds, rewards, obs = self.run_episode()
            all_rewards.append(rewards)
            all_obs.append(obs)
        all_rewards_df = pd.DataFrame(
            {f'Trial-{i}': all_rewards[i].reward for i in range(num_trials)},
            index=all_rewards[0].index
        )
        ec_cols = lambda df: df[[c for c in df.columns if 'energyconsumption' in c]]
        energyconsumption_df = pd.DataFrame(
            {f'Trial-{i}': ec_cols(all_obs[i]).sum(axis=1)
            for i in range(num_trials)},
            index=all_obs[0].index
        )
        mean_rewards = all_rewards_df.mean(axis=1)
        rewards_lower_std = mean_rewards - all_rewards_df.std(axis=1)
        rewards_upper_std = mean_rewards + all_rewards_df.std(axis=1)
        self.plot_speeds_and_rewards(
            speeds, mean_rewards,
            rewards_lower_std=rewards_lower_std,
            rewards_upper_std=rewards_upper_std,
            rewards_global_min=all_rewards_df.min(axis=None),
            output_file=output_file
        )
        return all_rewards_df, energyconsumption_df

def plot_training_rewards(log_file, output_file=None):
    log = pd.read_csv(log_file)
    rewards = log['rollout/ep_rew_mean']
    timesteps = log['time/total_timesteps']
    plt.plot(timesteps, rewards)
    plt.ylabel('reward (averaged per rollout)')
    plt.xlabel('timestep')
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()


if __name__=='__main__':
    results_dir = '../Results/No_Uncertainty/Anytown_Modified'
    model_nr = '15'
    log_dir = path.join(results_dir, f'model_{model_nr}_log')
    objective_weights_file = (
        '../Data/Parameters_for_Optimization/objective_weights_5.yaml'
    )
    make_speed_aware = lambda x: SpeedAwareObservation(x, n_timesteps=3)
    add_previous_obs = lambda x: FrameStackObservation(x, stack_size=2)
    rescale_action = lambda x: RescaleAction(x, -1, 1)
    make_time_aware = lambda x: EpisodeStepsEncoder(x)
    env = make_vec_env(
        env_id = ATM_env,
        env_kwargs = dict(
            max_pump_speed=1,
            objective_weights_file=objective_weights_file
        ),
        seed = 42,
        wrapper_class = lambda x: add_previous_obs(NormalizeHydraulics(rescale_action(x))),
        vec_env_cls = DummyVecEnv
    )
#    env = leakdb_env(scenario_nrs=[302], max_pump_speed=1, objective_weights_file=objective_weights_file)
#    env = make_time_aware(NormalizeHydraulics(rescale_action(env)))
    model_path = path.join(log_dir, 'best_model.zip')
    model = SAC('MlpPolicy', env, seed=5563)
    model.set_parameters(model_path)
#    model = ConstantEqualAgent(env, 0.6)
    agent_analyzer = AgentAnalyzer(env, model)
    speeds, rewards, obs = agent_analyzer.run_episode()
    rewards.to_csv(path.join(results_dir, f'rewards_{model_nr}.csv'))
    obs.to_csv(path.join(results_dir, f'obs_{model_nr}.csv'))
    agent_analyzer.plot_speeds_and_rewards(speeds, rewards.loc[:, ['reward', 'pressure_obj', 'pump_price_obj']], output_file = path.join(results_dir, f'model_{model_nr}_performance.png'))
    agent_analyzer.close()
    plt.show()

