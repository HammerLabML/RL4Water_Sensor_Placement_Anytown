import matplotlib.pyplot as plt
import pandas as pd

csv_files = ['patterns_ATM_old.csv', 'patterns_ATM_new.csv']
old_df, new_df = [pd.read_csv(csv_file, index_col='Time') for csv_file in csv_files]
fig, ax = plt.subplots()
old_df['DEM'].plot(label='old pattern', ax=ax)
new_df['DEM'].plot(label='new pattern', ax=ax)
plt.ylabel('Demand Multiplier')
plt.legend()
plt.savefig('../../Intermediate_Results_Anytown_Modified/old_vs_new_pattern.png')

