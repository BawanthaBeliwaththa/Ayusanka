import pandas as pd

df = pd.read_csv("global_health_estimates_life_expectancy_and_leading_causes_of_death_and_disability_indicators_lk.csv")

le_codes = ["WHOSIS_000001", "WHOSIS_000002"]
le_df = df[df['GHO (CODE)'].isin(le_codes)]

print("Unique Life Expectancy Indicators:")
print(le_df[['GHO (CODE)', 'GHO (DISPLAY)']].drop_duplicates())

print("\nNumber of records:", len(le_df))

if len(le_df) > 0:
    print("\nSexes and Dimensions in Life Expectancy data:")
    print(le_df['DIMENSION (NAME)'].value_counts())
    
    print("\nYears covered for Life Expectancy:")
    print(sorted(le_df['YEAR (DISPLAY)'].unique()))
    
    # Pivot for sample
    sample_df = le_df[le_df['GHO (CODE)'] == 'WHOSIS_000001']
    sample_pivot = sample_df.pivot_table(index="YEAR (DISPLAY)", columns="DIMENSION (NAME)", values="Numeric", aggfunc='first')
    print("\nLife Expectancy at Birth (WHOSIS_000001) over the years (Sample):")
    print(sample_pivot.sort_index(ascending=False).head(15))
