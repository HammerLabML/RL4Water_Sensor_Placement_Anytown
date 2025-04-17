import os
from os import path
import sys
import glob
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator
from tensorboardX import SummaryWriter
import numpy as np

def load_scalars(log_dir):
    """
    Load TensorBoard files and return their contents as a dict.

    @param log_dir, str, the directory containing TensorBoard file(s)
    @returns a defaultdict(dict) (i.e. a dict that returns empty dicts for all
    non-initialized keys)
      keys: tag names of the tensorboard (e.g. 'rollout/ep_rew_mean')
      values: dictionaries mapping time steps to recorded values
    """
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    data = defaultdict(dict)  # {tag: {step: value}}
    
    for tag in ea.Tags()['scalars']:
        for scalar_event in ea.Scalars(tag):
            data[tag][scalar_event.step] = scalar_event.value
    return data

def merge_scalar_dicts(dicts):
    """
    Average multiple TensorBoard results

    @param dicts, list of defaultdict as returned by load_scalars (see doc there)
    @returns a defaultdict(dict) containing the averaged results of the inputs
      keys: tag names (e.g. 'rollout/ep_rew_mean')
      values: a dict containing the averaged values for each time step
    """
    merged = defaultdict(lambda: defaultdict(list))  # {tag: {step: [values]}}

    for d in dicts:
        for tag, steps in d.items():
            for step, value in steps.items():
                merged[tag][step].append(value)

    averaged = defaultdict(dict)
    for tag, steps in merged.items():
        for step, values in steps.items():
            averaged[tag][step] = np.mean(values)

    return averaged

def write_averaged_log(averaged_data, output_dir):
    """
    Write a TensorBoard file containing results to an output directory.

    @param averaged_data, defaultdict(dict) like returned by average_scalar_dicts
      keys: tag names (e.g. 'rollout/ep_rew_mean')
      values: dictionaries containing recorded values per timestep. The values
        should be averaged with average_scalar_dicts before calling this method
    @param output_dir, str, name of the directory to write the results to
    @return This method writes a TensorBoard file and has no return value
    """
    writer = SummaryWriter(log_dir=output_dir)
    for tag, steps in averaged_data.items():
        for step in sorted(steps):
            writer.add_scalar(tag, steps[step], step)
    writer.close()

def average_tensorboard_logs(log_dir):
    """
    Average over all tensorboard logs in a directory and write a summary folder.

    @param log_dir, str
    The directory should have the following structure (as will be automatically the
    case when logs were created using the cw2 package):
    - sub-folders of the form rep_00, rep_01, ... should exist. In cw2, these
      represent repetitions with different random seeds
    - Each sub-folder should have one TensorBoard file and the TesnroBoard
      files should contain the same tags and the same logging interval across
      folders
      WARNING: If multiple TensorBoard files are present, they will all be included,
      so make sure you remove files from spurious runs
    @returns This method has no return value
    Instead, a new folder 'summary' is created in log_dir that contains a
    TensorBoard file with the averaged results
    """
    run_dirs = [
        path.join(log_dir, d) for d in os.listdir(log_dir)
        if path.isdir(path.join(log_dir, d))
        and d.startswith('rep_')
    ]
    run_dirs = list(sorted(run_dirs, key=lambda d: int(d.split('_')[-1])))
    breakpoint()
    all_data = [load_scalars(d) for d in run_dirs]
    averaged = merge_scalar_dicts(all_data)
    output_dir = path.join(log_dir, 'summary')
    write_averaged_log(averaged, output_dir)

if __name__=='__main__':
    if len(sys.argv) < 2:
        print(
            f'Please provide the path to a directory containing multiple runs'
            f'as a command-line argument.'
        )
        exit()
    else:
        log_dir = sys.argv[1]
        average_tensorboard_logs(log_dir)

