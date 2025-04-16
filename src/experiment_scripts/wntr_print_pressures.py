import wntr

inp_file = 'net1.inp'
wn = wntr.network.WaterNetworkModel(inp_file)
sim = wntr.sim.WNTRSimulator(wn)
res = sim.run_sim()
pressures = res.node['pressure']
print('Computed Pressures using WNTR')
print(pressures.head())

