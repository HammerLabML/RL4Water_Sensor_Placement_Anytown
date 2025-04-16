import wntr
import sys
import pandas as pd

if len(sys.argv) < 3:
    print(
        f'Please provide the following command-line arguments:\n'
        f'- an .inp-file for demand parsing\n'
        f'- an output file to which the patterns are written in .csv-format'
    )
    exit()

inp_file = sys.argv[1]
wn = wntr.network.WaterNetworkModel(inp_file)
time_options = wn.options.time
duration = int(time_options.duration)
report = int(time_options.report_timestep)
timesteps = range(0, duration + 1, report)
res = pd.DataFrame(index=timesteps, columns=wn.pattern_name_list)
for pattern_name in wn.pattern_name_list:
    pattern = wn.get_pattern(pattern_name)
    for timestep in timesteps:
        res.loc[timestep, pattern_name] = pattern.at(timestep)
res.index.name = 'Time'
output_file = sys.argv[2]
res.to_csv(output_file)

