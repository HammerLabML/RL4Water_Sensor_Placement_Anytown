from stable_baselines3.common.utils import obs_as_tensor
import numpy as np
import torch

device = 'cpu'

def qvalues(model, obs):
    obs = obs[np.newaxis, ...]
    obs_tensor = obs_as_tensor(obs, device=device)
    with torch.no_grad():
        res = model.q_net(obs_tensor)
    return np.array(res).flatten()

