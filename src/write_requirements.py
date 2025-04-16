import argparse
import gymnasium as gym
import matplotlib
import numpy as np
import pandas
import yaml
import scipy
import stable_baselines3 as sb3
import rl_zoo3 as rz3
import torch
import subprocess

content = (
    f'argparse>={argparse.__version__}\n'
    f'gymnasium>={gym.__version__},<1.0\n'
    f'matplotlib>={matplotlib.__version__}\n'
    f'numpy>={np.__version__}\n'
    f'pandas>={pandas.__version__}\n'
    f'pyyaml>={yaml.__version__}\n'
    f'python>=3.10\n' # hard-coed Python version
    f'scipy>={scipy.__version__}\n'
    f'stable_baselines3>={sb3.__version__}\n'
    f'rl_zoo3>={rz3.__version__}\n'
    f'torch>={torch.__version__}\n'
)
epyt_flow_version_file = '/Users/paulstahlhofen/mambaforge/envs/rl4water/lib/python3.12/site-packages/epyt_flow/VERSION'
with open(epyt_flow_version_file, 'r') as fp:
    epyt_flow_version = fp.read().strip('\n')
content += f'epyt_flow>={epyt_flow_version}\n'
p = subprocess.Popen(['tensorboard', '--version'], stdout=subprocess.PIPE)
output, error = p.communicate()
tensorboard_version = output.decode().strip('\n')
content += f'tensorboard>={tensorboard_version}\n'

filename = 'REQUIREMENTS.TXT'
with open(filename, 'w') as fp:
    fp.write(content)

