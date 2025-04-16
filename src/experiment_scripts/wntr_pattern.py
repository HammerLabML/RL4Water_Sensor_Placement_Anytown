import sys
main_path = '/Users/paulstahlhofen/Documents/Water_Futures/RL4Water/src'
sys.path.insert(1, main_path)
import os
os.chdir('..')

import wntr

names = lambda obj: [n for n in dir(obj) if n[0]!='_']
mshow = lambda obj, m: [method for method in dir(obj) if m in method]

inp_file = '../Data/Anytown/ATM.inp'
wn = wntr.network.WaterNetworkModel(inp_file)

