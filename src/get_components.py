import gymnasium
import stable_baselines3.common.base_class
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from normalize_hydraulics import NormalizeHydraulics
from speed_aware_observation import SpeedAwareObservation
from gymnasium.wrappers import RescaleAction, TimeAwareObservation, FrameStackObservation, TimeLimit

from sensor_selection import SensorSelectionWrapper, SensorPlacementGenerator
from time_encoders import WeekdayEncoder, TimeOfDayEncoder, EpisodeStepsEncoder


def get_algorithm(config, env, model_path=None, seed=42) -> stable_baselines3.common.base_class.BaseAlgorithm:
    if config['name'] == 'SAC':
        from stable_baselines3 import SAC
        model = SAC(
            config['policy'], env, ent_coef=config['ent_coef'],
            gamma=config['gamma'], learning_rate=float(config['learning_rate']),
            batch_size=config['batch_size'],tau=float(config['tau']),
            verbose=1, seed=seed
        )
    elif config['name'] == 'PPO':
        from stable_baselines3 import PPO
        model = PPO(
            config['policy'], env,
            verbose=1, seed=seed
        )
    elif config['name'] == 'constant_equal':
        from dummy_agents import ConstantEqualAgent
        model = ConstantEqualAgent(env, config['speed'])
    elif config['name'] == 'recorded':
        from dummy_agents import RecordedAgent
        model = RecordedAgent.from_file(env, model_path)
    else:
        raise NotImplementedError("Algorithm '{}' not implemented.".format(config['params']['algorithm']['name']))

    if model_path is not None and config['name']!='recorded':
        model.set_parameters(model_path)
    return model


def get_wrapper(env, wrapper_names=None, sensor_file=None, obs_filter=['flow','pump'], wrapper_seed=0):

    wrapper_dict = {
    'make_speed_aware': lambda x: SpeedAwareObservation(x, n_timesteps=3),
    'add_previous_obs' : lambda x: FrameStackObservation(x, stack_size=2),
    'add_time_limit' : lambda x: TimeLimit(x, x.unwrapped._max_itr_till_truncation),
    'make_time_aware' : lambda x: EpisodeStepsEncoder(x),
    'rescale_action' : lambda x: RescaleAction(x, -1, 1),
    'normalize_hydraulics' : lambda x: NormalizeHydraulics(x),
    'time_of_day': lambda  x: TimeOfDayEncoder(x),
    'weekday': lambda x: WeekdayEncoder(x),
    'pressure_sensors': lambda x: SensorSelectionWrapper(x, sensor_file, obs_filter, wrapper_seed)

    }
    for wrapper in reversed(wrapper_names):
        env = wrapper_dict[wrapper](env)
    return env

def get_env(config, n_envs, log_dir, seed=42) -> gymnasium.Env:
    standard_sim_options = dict()
    if 'standard_sim_options' in config.keys():
        standard_sim_options.update(config['standard_sim_options'])
    env_kwargs = dict(
        max_pump_speed=config['max_pump_speed'],
        min_pump_speed=config['min_pump_speed'],
        max_speed_change=config['max_speed_change'],
        objective_weights_file=config['objective_weights_file'],
        model_uncertainty_file=config['model_uncertainty_file'],
        standard_sim_options=standard_sim_options
    )
    if config['env_id'] == 'net1_env':
        from pump_control_envs import net1_env
        env_id = net1_env
    elif config['env_id'] == 'leakdb_env':
        from pump_control_envs import leakdb_env
        env_id = leakdb_env
        env_kwargs.update(
            dict(
                start_scenario=config['start_scenario'],
                end_scenario=config['end_scenario'],
            )
        )
    elif config['env_id'] == 'ATM_env':
        from pump_control_envs import ATM_env
        env_id = ATM_env
        env_kwargs['version'] = config['version']
    elif config['env_id'] == 'ky16_env':
        from pump_control_envs import KY16_env
        env_id = KY16_env
    elif config['env_id'] == 'nj1_env':
        from pump_control_envs import NJ1_env
        env_id = NJ1_env
    elif config['env_id'] == 'net3_env':
        from pump_control_envs import net3_env
        env_id = net3_env
    elif config['env_id'] == 'dtown_env':
        from pump_control_envs import Dtown_env
        env_id = Dtown_env
    else:
        raise NotImplementedError("Environment '{}' not implemented.".format(config['env_id']))

    if n_envs > 1:
        env = make_vec_env(
            env_id=env_id,
            env_kwargs=env_kwargs,
            n_envs=config['n_envs'],
            seed=seed,
            monitor_dir=log_dir,
            wrapper_class=get_wrapper,
            wrapper_kwargs=config['wrapper'],
            vec_env_cls=SubprocVecEnv
        )
    elif n_envs == 1:
        env = make_vec_env(
            env_id=env_id,
            env_kwargs=env_kwargs,
            seed=seed,
            monitor_dir=log_dir,
            wrapper_class=get_wrapper,
            wrapper_kwargs=config['wrapper'],
            vec_env_cls=DummyVecEnv
        )
    else:
        raise ValueError('n_envs must be a positive integer.')
    return env
