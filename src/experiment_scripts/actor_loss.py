import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from os import path
import pandas as pd
import matplotlib.pyplot as plt

results_dir = '../Results/No_Uncertainty/Anytown'
progress_files = [path.join(results_dir, subdir, 'progress.csv') for subdir in [f'model_{i}_log' for i in [6,7,9,10,11,12,13]]]
progress_files.extend([path.join(results_dir, 'model_8_log', filename) for filename in ['progress_1-50K.csv', 'progress_50K-200K.csv']])
shortened_paths = [path.join(path.basename(path.dirname(progress_file)), path.basename(progress_file)) for progress_file in progress_files]

progress_dfs = {shortened_path: pd.read_csv(progress_file) for (shortened_path, progress_file) in zip(shortened_paths, progress_files)}
for shortened_path, progress_df in progress_dfs.items():
    last_actor_loss = progress_df['train/actor_loss'].tail(1).item()
    print(f'{shortened_path}: {last_actor_loss:.3f}')

#for shortened_path, progress_df in progress_dfs.items():
#    fig, ax = plt.subplots()
#    progress_df['train/actor_loss'].plot()
#    fig.suptitle(shortened_path)
#plt.show()
