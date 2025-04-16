# Optimizing pumps in the first timestep with standard optimizers

This is done to check if it is in principle possible to find a pump setting
for a given network that satisfies the pressure constraints at all nodes.

Note: If `initial_tank_levels` are not given in the output files, they are
eqaul to the initial tank levels specified in the respective inp-files.
Changing and logging of initial tank levels was implemented later to check if
pressure constraints can be satisfied easier if tanks are at their lowest
possible level initially.
