import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from os import path
import pandas as pd
import matplotlib.pyplot as plt

results_dir = '../Results/No_Uncertainty/Anytown'
model_nr = '15'
model_dirname = f'model_{model_nr}_log'
progress_file = path.join(results_dir, model_dirname, 'progress.csv')
progress_df = pd.read_csv(progress_file)
progress_df.rename(
    columns={
        'rollout/ep_rew_mean': 'rew',
        'train/ent_coef_loss': 'el',
        'train/critic_loss': 'cl',
        'train/critic_grad_norm': 'cg',
        'train/actor_loss': 'al',
        'train/ent_coef': 'ec',
        'train/n_updates': 'nu'
    },
    inplace=True
)
progress_df.rew += 50
progress_df.cg /= 10
progress_df.iloc[10:70, :].loc[:, ['cg', 'cl']].plot()
plt.show()
#progress_df.loc[progress_df.cl>60, 'cl'] = 60
#progress_df.ec *= 80
#progress_df.nu /= 4_000
#progress_df.rew -= progress_df.rew.min()
#progress_df.rew /= progress_df.rew.max()
#progress_df.cl /= 60
#progress_df.cg /= 25
#max_rew_idx = progress_df.rew.argmax()
#window_size = 20
#low_idx, high_idx = max_rew_idx - window_size, max_rew_idx + window_size
#progress_df.iloc[low_idx:high_idx, :].loc[:, ['rew', 'cl', 'cg']].plot()
#plt.axvline(max_rew_idx)
#plt.show()
#
