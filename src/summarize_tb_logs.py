import os
from os import path
import sys
import glob
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator
from tensorboardX import SummaryWriter
import numpy as np

def load_scalars(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    data = defaultdict(dict)  # {tag: {step: value}}
    
    for tag in ea.Tags()['scalars']:
        for scalar_event in ea.Scalars(tag):
            data[tag][scalar_event.step] = scalar_event.value
    return data

def merge_scalar_dicts(dicts):
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
    writer = SummaryWriter(log_dir=output_dir)
    for tag, steps in averaged_data.items():
        for step in sorted(steps):
            writer.add_scalar(tag, steps[step], step)
    writer.close()

def average_tensorboard_logs(log_dir):
    run_dirs = [
        path.join(log_dir, d) for d in os.listdir(log_dir)
        if path.isdir(path.join(log_dir, d))
        and d.startswith('rep_')
    ]
    run_dirs = list(sorted(run_dirs, key=lambda d: int(d.split('_')[-1])))
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

