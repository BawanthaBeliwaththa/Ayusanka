import pandas as pd
import json

print("=== LE & HALE & NCD ===")
try:
    df_le = pd.read_csv("global_health_estimates_life_expectancy_and_leading_causes_of_death_and_disability_indicators_lk.csv")
    print("LE columns:", list(df_le.columns))
    print("LE indicators:", df_le['GHO (CODE)'].unique())
    print("LE years:", sorted(df_le['YEAR (DISPLAY)'].unique()))
except Exception as e:
    print("Error LE:", e)

print("\n=== Air Pollution ===")
try:
    df_air = pd.read_csv("air_pollution_indicators_lka.csv")
    print("Air indicators:", df_air['GHO (CODE)'].unique())
    print("Air years:", sorted(df_air['YEAR (DISPLAY)'].unique()))
except Exception as e:
    print("Error Air:", e)

print("\n=== Alcohol ===")
try:
    with open("who_lk_recorded_per_capita.json", "r") as f:
        alc = json.load(f)
    years = sorted(list(set(r.get("TimeDim") for r in alc if r.get("Dim1") == "ALCOHOLTYPE_SA_TOTAL")))
    print("Alcohol years:", years)
except Exception as e:
    print("Error Alc:", e)

print("\n=== Tobacco ===")
try:
    with open("who_lk_fetched_data.json", "r") as f:
        tob = json.load(f)
    tob_years = sorted(list(set(r.get("TimeDim") for r in tob.get("tobacco_prevalence", []))))
    cig_years = sorted(list(set(r.get("TimeDim") for r in tob.get("cig_prevalence", []))))
    print("Tobacco years:", tob_years)
    print("Cigarette years:", cig_years)
except Exception as e:
    print("Error Tob:", e)
