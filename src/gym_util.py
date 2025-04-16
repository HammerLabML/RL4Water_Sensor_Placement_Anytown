import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete
from gymnasium.wrappers import NormalizeObservation
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
import numpy as np
import pickle

def to_multidiscrete(elem, multidiscrete):
    """
    Convert the element of a Discrete space
    to the equivalent element of a MultiDiscrete space.
    """
    dimensions = multidiscrete.nvec
    discrete = Discrete(dimensions.prod())
    if not discrete.contains(elem):
        raise ValueError(f"{discrete} does not contain {elem}.")
    ndim = len(dimensions)
    remaining_prods = np.array(
        [dimensions[i+1:].prod() for i in range(ndim)],
        dtype=multidiscrete.dtype
    )
    res = []
    for remaining_prod, dim in zip(remaining_prods, dimensions):
        res.append(elem // remaining_prod)
        elem = elem % remaining_prod
    res = np.array(res, dtype=multidiscrete.dtype)
    return res

def to_discrete(elem, multidiscrete):
    """
    Convert the element of a MultiDiscrete space
    to an equivalent element of a Discrete space.
    """
    if not multidiscrete.contains(elem):
        raise ValueError(f"{multidiscrete} does not contain {elem}.")
    dimensions = multidiscrete.nvec
    ndim = len(dimensions)
    remaining_prods = np.array(
        [dimensions[i+1:].prod() for i in range(ndim)],
        dtype=multidiscrete.dtype
    )
    return np.inner(elem, remaining_prods)

def save_obs_norm_stats(env, output_file):
    if isinstance(env, VecEnv):
        if len(env.envs)==1:
            env = env.envs[0]
        else:
            raise ValueError(
                f'You passed a vectorized environment with multiple elements. '
                f'Please choose one of them for which normalization statistics '
                f'sould be saved.'
            )
    while not isinstance(env, NormalizeObservation):
        try:
            env = env.env
        except AttributeError as e:
            print(
                f'ERROR: The environment was never wrapped with a '
                f'NormalizeObservation wrapper. No statistics can be saved.'
            )
            raise e
    stats = {
        'count': env.obs_rms.count,
        'mean': env.obs_rms.mean,
        'var': env.obs_rms.var
    }
    with open(output_file, 'wb') as fp:
        pickle.dump(stats, fp)

def set_obs_norm_stats(normalize_observation_wrapper, stats_file):
    with open(stats_file, 'rb') as fp:
        stats = pickle.load(fp)
    normalize_observation_wrapper.obs_rms.count = stats['count']
    normalize_observation_wrapper.obs_rms.mean = stats['mean']
    normalize_observation_wrapper.obs_rms.var = stats['var']
    return normalize_observation_wrapper

def basic_env_is_wrapped_with(env, wrapper_class):
    """
    Check if a Gymnasium environment is wrapped with a specific wrapper class.

    Parameters:
    - env: The Gymnasium environment to check.
    - wrapper_class: The wrapper class to look for.

    Returns:
    - bool: True if the environment is wrapped with the specified wrapper class
    """
    current_env = env
    while isinstance(current_env, gym.Wrapper):
        if isinstance(current_env, wrapper_class):
            return True
        current_env = current_env.env  # Go to the inner wrapped environment
    return False

def vec_env_is_wrapped_with(venv, wrapper_class):
    current_venv = venv
    while isinstance(current_venv, VecEnvWrapper):
        if isinstance(current_venv, wrapper_class):
            return True
        current_venv = current_venv.venv
    return False

def is_wrapped_with(env, wrapper_class):
    if isinstance(env, gym.Env):
        return basic_env_is_wrapped_with(env, wrapper_class)
    elif isinstance(env, VecEnv):
        return vec_env_is_wrapped_with(env, wrapper_class)
    else:
        raise ValueError(
            f'env must be either of type gymnasium.Env '
            f'or of type stalbe_baselines3.common.vec_env.base_vec_env.VecEnv.'
        )

def inner_env_is_wrapped_with(venv, wrapper_class):
    l
if __name__=='__main__':
    multidiscrete = MultiDiscrete([3, 4, 2])
    md_elems = []
    for i in range(24):
        md_elem = np.array(to_multidiscrete(i, multidiscrete))
        md_elems.append(md_elem)
        print(f"{i} -> {md_elem}")
    print()
    print('='*30)
    print()
    for md_elem in md_elems:
        d_elem = to_discrete(md_elem, multidiscrete)
        print(f"{md_elem} -> {d_elem}")

