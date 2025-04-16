from itertools import groupby
import wntr

def successive_duplicates(lst):
    uniq_with_count = [(key, sum(1 for _ in group)) for key, group in groupby(lst)]
    keys, counts = zip(*uniq_with_count, strict=True)
    if all([c==counts[0] for c in counts]):
        return list(keys), counts[0]
    else:
        return lst, 0

def interpolate_linearly(demand_pattern, old_timestep, new_timestep):
    demand_pattern, n_successive_duplicates = successive_duplicates(demand_pattern)
    if n_successive_duplicates:
        old_timestep *= n_successive_duplicates
    # Due to the cyclic nature of patterns,
    # their beginning should be reached in the end
    demand_pattern.append(demand_pattern[0])
    linear_from_to = lambda s, start, stop: start + (s/old_timestep) * (stop-start)
    piecewise_linear = lambda s: linear_from_to(
        s % old_timestep,
        demand_pattern[s // old_timestep],
        demand_pattern[s // old_timestep + 1]
    )
    total_duration = (len(demand_pattern)-1) * old_timestep
    new_demand_pattern = [
        piecewise_linear(s) for s in range(0, total_duration, new_timestep)
    ]
    return new_demand_pattern

def update_anytown_patterns():
    inp_file = '../Data/Anytown/ATM.inp'
    wn = wntr.network.WaterNetworkModel(inp_file)
    demand_patterns = [
        wn.get_pattern(pat) for pat in wn.pattern_name_list
        if pat.startswith('DEM')
    ]
    for demand_pattern in demand_patterns:
        old_multipliers = list(demand_pattern.multipliers)
        new_multipliers = interpolate_linearly(old_multipliers, 3600, 300)
        demand_pattern.multipliers = new_multipliers
    wn.write_inpfile('../Data/Anytown/ATM_adapted.inp')

update_anytown_patterns()

