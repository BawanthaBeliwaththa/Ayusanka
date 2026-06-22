import pandas as pd
import json

df_le_local = pd.read_csv("global_health_estimates_life_expectancy_and_leading_causes_of_death_and_disability_indicators_lk.csv")
df_air_local = pd.read_csv("air_pollution_indicators_lka.csv")

with open("who_lk_recorded_per_capita.json", "r") as f:
    alcohol_records = json.load(f)

with open("who_lk_fetched_data.json", "r") as f:
    tobacco_records_all = json.load(f)

print("--- Life Expectancy at Birth (WHOSIS_000001) years ---")
print(sorted(df_le_local[df_le_local['GHO (CODE)'] == 'WHOSIS_000001']['YEAR (DISPLAY)'].unique()))

print("--- Healthy Life Expectancy (WHOSIS_000002) years ---")
print(sorted(df_le_local[df_le_local['GHO (CODE)'] == 'WHOSIS_000002']['YEAR (DISPLAY)'].unique()))

print("--- NCD Mortality (NCDMORT3070) years ---")
print(sorted(df_le_local[df_le_local['GHO (CODE)'] == 'NCDMORT3070']['YEAR (DISPLAY)'].unique()))

print("--- PM2.5 (SDGPM25) years ---")
print(sorted(df_air_local[df_air_local['GHO (CODE)'] == 'SDGPM25']['YEAR (DISPLAY)'].unique()))

print("--- Alcohol years ---")
alc_years = sorted(list(set(r.get("TimeDim") for r in alcohol_records if r.get("Dim1") == "ALCOHOLTYPE_SA_TOTAL")))
print(alc_years)
