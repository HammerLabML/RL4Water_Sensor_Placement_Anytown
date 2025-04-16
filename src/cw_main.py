# MY_CW_MAIN.py
from cw2 import experiment, cw_error
from cw2.cw_data import cw_logging
from cw2 import cluster_work

import stable_baselines3.common as common

import os
from os import path
import numpy as np

from save_best_model_callback import SaveBestModelCallback

from get_components import get_algorithm, get_env
from analyze_agent import AgentAnalyzer
from eval_callback import CustomEvalCallback


class MyExperiment(experiment.AbstractExperiment):
    # ...

    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        self.train = config['params']['train']
        config['params']['env']['wrapper']['wrapper_seed'] = rep
        load_existing_model = 'path' in config['params']['algorithm'].keys()
        constant_agent = 'constant' in config['params']['algorithm']['name']
        if not self.train and not load_existing_model and not constant_agent:
            raise RuntimeError(
                f'You must either train a new model, '
                f'load an existing one or supply a constant agent.'
            )
        self.results_dir = config['_rep_log_path']

        os.makedirs(self.results_dir, exist_ok=True)
        env_params = config['params']['env']
        seed = rep * (env_params['n_envs'] + 1)

        self.analyze = config['params']['analyze_agent']
        self.analysis_kwargs = dict()
        if 'analysis_kwargs' in config['params'].keys():
            self.analysis_kwargs.update(config['params']['analysis_kwargs'])

        self.env = get_env(env_params, env_params['n_envs'], self.results_dir, seed=seed)
        if self.analyze:
            test_env_params = env_params.copy()
            if 'test_env' in config['params'].keys():
                test_env_params.update(config['params']['test_env'])
            self.test_env = get_env(
                test_env_params,
                n_envs=1,
                log_dir=None,
                seed=seed + test_env_params['n_envs']
            )
        model_path = None
        if load_existing_model:
            if config['params']['algorithm']['path']=='self':
                model_path = path.join(
                    self.results_dir, 'best_model.zip'
                )
            else:
                model_path = config['params']['algorithm']['path']
        self.model = get_algorithm(
            config['params']['algorithm'], self.env,
            model_path=model_path, seed=seed
        )

        new_logger = common.logger.configure(self.results_dir, ['stdout', 'csv', 'tensorboard'])
        self.model.set_logger(new_logger)
        self.callback = [
            SaveBestModelCallback(
                check_freq=1000, log_dir=self.results_dir,
                save_norm_stats=False, verbose=False
            ),
            CustomEvalCallback(
                eval_env=self.test_env,
                eval_freq=100,
                reward_columns=config['params']['logging']['reward_columns'],
                verbose=1
            )
        ]

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        if self.train:
            # Perform your existing task
            self.model.learn(
                total_timesteps=config['iterations'],
                callback=self.callback
            )

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        if not crash:
            if self.train: # don't save the model again if it was just loaded
                self.model.save(self.results_dir)
            if self.analyze:
                print()
                print('='*30)
                print('Starting Evaluation')
                print('='*30)
                print()
                agent_analyzer = AgentAnalyzer(self.test_env, self.model)
                # A single episode run will suffice if there is no uncertainty
                # in the network
                if self.test_env.get_attr('unwrapped')[0].model_uncertainty is None:
                    res = agent_analyzer.run_episode(**self.analysis_kwargs)
                    if self.analysis_kwargs['return_actions']:
                        speeds, rewards, obs, actions = res
                        np.save(path.join(self.results_dir, 'actions.npy'), actions)
                    else:
                        speeds, rewards, obs = res
                    rewards.to_csv(path.join(self.results_dir, f'rewards.csv'))
                    obs.to_csv(path.join(self.results_dir, f'obs.csv'))
                    agent_analyzer.plot_speeds_and_rewards(
                        speeds,
                        rewards.loc[:, ['reward', 'pressure_obj', 'pump_price_obj']],
                        output_file=path.join(
                            self.results_dir, 'model_performance.png'
                        )
                    )
                else: # there is model uncertainty => compare multiple runs
                    all_rewards, energy_use = agent_analyzer.compare_test_episodes(
                        num_trials=10,
                        output_file=path.join(
                            self.results_dir, 'averaged_model_performance.png'
                        )
                    )
                    all_rewards.to_csv(
                        path.join(self.results_dir, 'all_rewards.csv')
                    )
                    energy_use.to_csv(
                        path.join(self.results_dir, 'energy_use.csv')
                    )
        self.env.close()
        self.test_env.close()

if __name__ == "__main__": # Give the MyExperiment Class, not MyExperiment() Object!!
    cw = cluster_work.ClusterWork(MyExperiment)

    # Optional: Add loggers
    #cw.add_logger(...)

    # RUN!
    cw.run()
