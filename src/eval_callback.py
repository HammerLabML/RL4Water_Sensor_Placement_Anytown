from stable_baselines3.common.callbacks import BaseCallback

from analyze_agent import AgentAnalyzer

from analyze_agent import AgentAnalyzer
import numpy as np

class CustomEvalCallback(BaseCallback):
    """
        A custom callback that derives from ``BaseCallback``.

        :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
        """

    def __init__(self, eval_env, eval_freq = 100, reward_columns='all', verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.reward_columns = reward_columns
        self.constraints = self.eval_env.envs[0].unwrapped.objective_calculator.network_constraints

    def log_mean_and_std(self, df, column):
        self.logger.record(f'eval/ep_{column}_mean', df[column].mean())
        self.logger.record(f'eval/ep_{column}_std', df[column].std())


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: If the callback returns False, training is aborted early.
        """
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            agent_analyzer = AgentAnalyzer(self.eval_env, self.model)
            speeds, reward_info, obs  = agent_analyzer.run_episode()

            ### get average and std speed for each pump and log them
            for column in speeds.columns:
                self.log_mean_and_std(speeds, column)

            ### get average and std of partial rewards and log them
            logged_reward_columns = (
                reward_info.columns if self.reward_columns=='all'
                else self.reward_columns
            )
            for column in logged_reward_columns:
                self.log_mean_and_std(reward_info, column)


            obs = obs[[col for col in obs if col.startswith('press')]]
            ## analyze obs for percentage of nodes above threshold
            nodes_above_threshold = np.mean(obs > self.constraints.max_pressure)
            self.logger.record("eval/ep_nodes_above_threshold", nodes_above_threshold)
            ## analyze obs for percentage of nodes below threshold
            nodes_below_threshold = np.mean(obs < self.constraints.min_pressure)
            self.logger.record("eval/ep_nodes_below_threshold", nodes_below_threshold)
            ## analyze obs for percentage of nodes below zero
            nodes_below_zero = np.mean(obs < 0)
            self.logger.record("eval/ep_nodes_below_zero", nodes_below_zero)
            ## analyze obs for steps with any nodes below zero
            steps_below_zero = np.any(obs < 0, axis=1).sum()
            self.logger.record("eval/ep_timesteps_below_zero", steps_below_zero)

        return True

