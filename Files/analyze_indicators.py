import pandas as pd

def analyze_csv(file_path):
    print(f"=== Analyzing {file_path} ===")
    try:
        df = pd.read_csv(file_path)
        print("Columns:", list(df.columns))
        print("Number of rows:", len(df))
        unique_ghos = df[['GHO (CODE)', 'GHO (DISPLAY)']].drop_duplicates()
        print("\nUnique GHO Indicators:")
        for idx, row in unique_ghos.iterrows():
            print(f"- {row['GHO (CODE)']}: {row['GHO (DISPLAY)']}")
        print("\nYears covered:", sorted(df['YEAR (DISPLAY)'].unique()))
    except Exception as e:
        print("Error:", e)
    print("\n" + "="*50 + "\n")

analyze_csv("global_health_estimates_life_expectancy_and_leading_causes_of_death_and_disability_indicators_lk.csv")
analyze_csv("air_pollution_indicators_lka.csv")
