from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import os
import gym_util

"""
Code from the stable baselines 3 documentation.
Doc Link: https://stable-baselines3.readthedocs.io
Tutorial Link with code: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/monitor_training.ipynb#scrollTo=pUWGZp3i9wyf
"""

class SaveBestModelCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param save_norm_stats: (bool) If True, the normalization statistics of
      the best model will also be saved. This can apply only for models with a
      single environment wrapped with a NormalizeObservation wrapper (See
      gymnasium.wrappers.NormalizeObservation). Saving is done via
      gym_util.save_obs_norm_stats. 
    :param verbose: (int)
    """

    def __init__(
            self, check_freq: int, log_dir: str, save_norm_stats: bool, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.model_save_path = os.path.join(log_dir, "best_model")
        self.save_norm_stats = save_norm_stats
        if save_norm_stats:
            self.stats_save_path = os.path.join(log_dir, 'best_model_norm_stats')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.model_save_path is not None:
            os.makedirs(self.model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - "
			f"Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.model_save_path}.zip")
                    self.model.save(self.model_save_path)
                    if self.save_norm_stats:
                        if self.verbose > 0:
                            print(
                                f"Saving model normalization statistics to "
                                f"{self.stats_save_path}"
                            )
                        gym_util.save_obs_norm_stats(
                            self.model.get_env(), self.stats_save_path
                        )
        return True

