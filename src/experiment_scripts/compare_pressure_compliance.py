import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from os import path
import pandas as pd
import matplotlib.pyplot as plt
from objective_calculator import NetworkConstraints

def compare_pressure_compliance(obs_files, network_constraints, file_labels=None, output_file=None):
    if file_labels is None:
        file_labels = {obs_file: obs_file for obs_file in obs_files}
    obs_dfs = {
        file_labels[obs_file]: pd.read_csv(obs_file, index_col='Time')
        for obs_file in obs_files
    }
    fig, ax = plt.subplots()
    for file_label, obs_df in obs_dfs.items():
        pressure_cols = [c for c in obs_df if 'pressure' in c]
        min_pressure = obs_df[pressure_cols].min(axis=1)
        min_pressure.plot(ax=ax, label=file_label)
    ax.axhline(network_constraints.min_pressure)
    plt.legend()
    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()

if __name__=='__main__':
    network_constraints_file = '../Data/Net1/net1_constraints.yaml'
    network_constraints = NetworkConstraints.from_yaml(network_constraints_file)
    results_dir = '../Results/Multi_Network/LeakDB'
    obs_filenames = ['obs_4.csv', 'obs_constant_06.csv']
    obs_files = [
        path.join(results_dir, obs_filename) for obs_filename in obs_filenames
    ]
    file_labels = {
        path.join(results_dir, 'obs_4.csv'): 'our model',
        path.join(results_dir, 'obs_constant_06.csv'): 'baseline'
    }
    output_file = path.join(results_dir, 'comparison_pres_model_4_constant_06.png')
    compare_pressure_compliance(
        obs_files, network_constraints,
        file_labels=file_labels,
        output_file=output_file
    )

