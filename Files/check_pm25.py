import pandas as pd

df = pd.read_csv("air_pollution_indicators_lka.csv")
pm_df = df[df['GHO (CODE)'] == 'SDGPM25']

print("PM2.5 Indicators available:")
print(f"Total PM2.5 records: {len(pm_df)}")
print("\nUnique Residence Area Types:")
print(pm_df['DIMENSION (NAME)'].value_counts())
print("\nYears covered for PM2.5:")
print(sorted(pm_df['YEAR (DISPLAY)'].unique()))

print("\nPM2.5 concentrations over years (Sample):")
pivot_df = pm_df.pivot_table(index="YEAR (DISPLAY)", columns="DIMENSION (NAME)", values="Numeric", aggfunc='first')
print(pivot_df.sort_index(ascending=False).head(15))
