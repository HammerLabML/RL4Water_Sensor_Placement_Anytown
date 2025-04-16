import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

from time_encoders import WeekdayEncoder, TimeOfDayEncoder, EpisodeStepsEncoder
from pump_control_envs import leakdb_env
from analyze_agent import AgentAnalyzer
from dummy_agents import ConstantEqualAgent
import pandas as pd
import matplotlib.pyplot as plt

env = leakdb_env(scenario_nrs=[1])
env = TimeOfDayEncoder(WeekdayEncoder(env))
agent = ConstantEqualAgent(env, 0.6)
agent_analyzer = AgentAnalyzer(env, agent)
_, _, obs = agent_analyzer.run_episode()
agent_analyzer.close()
fig, ax = plt.subplots(1, 2)
weekday_cols = [c for c in obs.columns if 'is_' in c]
daytime_cols = [c for c in obs.columns if 'daytime' in c]
weekday_markers = pd.Series(list(range(7)), index=weekday_cols)
(obs[weekday_cols]*weekday_markers).sum(axis=1).plot(ax=ax[0])
obs.loc[:24*3600, daytime_cols].plot(ax=ax[1])
plt.show()

