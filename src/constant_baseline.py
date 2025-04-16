from os import path
import numpy as np
import pandas as pd

from dummy_agents import ConstantEqualAgent
from analyze_agent import AgentAnalyzer
from gymnasium import Env
from pump_control_envs import PumpControlEnv, ATM_env, net1_env, leakdb_env

def line_search_constant_equal_speed(env, search_interval=0.05):
    if not isinstance(env, Env):
        raise ValueError('env must be a gymnasium.Env')
    if not isinstance(env.unwrapped, PumpControlEnv):
        raise ValueError('env must be a PumpControlEnv')
    search_space = np.arange(
        0, env.unwrapped.max_pump_speed, search_interval,
        dtype=env.unwrapped.float_type
    )
    search_space[0] = 1e-10 # Avoid problems with zero speed
    results_df = pd.DataFrame(
        index=np.round(search_space, decimals=2),
        columns=['mean_reward', 'min_reward']
    )
    for constant_speed in search_space:
        agent = ConstantEqualAgent(env, constant_speed)
        agent_analyzer = AgentAnalyzer(env, agent)
        speeds, rewards, episode_obs = agent_analyzer.run_episode(keep_scenario=True)
        res = dict(
            mean_reward=rewards['reward'].mean(),
            min_reward=rewards['reward'].min()
        )
        results_df.loc[np.round(constant_speed, decimals=2)] = res
    env.close()
    return results_df

if __name__=='__main__':
    np.random.seed(42)
    scenario_nrs = np.random.choice(np.arange(301, 1001), size=10, replace=False)
    env = leakdb_env(scenario_nrs=scenario_nrs)
    total_results = dict()
    for scenario_nr in scenario_nrs:
        scenario_results = line_search_constant_equal_speed(env)
        total_results[scenario_nr] = scenario_results['mean_reward']
        env.reset(keep_scenario=False) # get the next scenario
    total_results = pd.DataFrame(total_results)
    total_results.index.name = 'speed'
    total_results.to_csv(
        '../Results/Multi_Network/LeakDB/constant_equal_speed_baseline.csv'
    )

