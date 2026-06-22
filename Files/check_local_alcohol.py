import pandas as pd

df = pd.read_csv("air_pollution_indicators_lka.csv")
alcohol_mask = df['GHO (CODE)'].str.contains("SA_", na=False)
alcohol_df = df[alcohol_mask]
print("Unique alcohol GHOs in local air pollution file:")
print(alcohol_df[['GHO (CODE)', 'GHO (DISPLAY)']].drop_duplicates())
print(f"\nNumber of alcohol records: {len(alcohol_df)}")
if len(alcohol_df) > 0:
    print("\nYears covered for alcohol:")
    print(sorted(alcohol_df['YEAR (DISPLAY)'].unique()))
    print("\nSample records:")
    print(alcohol_df[['GHO (CODE)', 'YEAR (DISPLAY)', 'DIMENSION (NAME)', 'Numeric', 'Value']].head(10))
