import pandas as pd

def search_terms(file_path):
    print(f"=== Searching {file_path} ===")
    try:
        df = pd.read_csv(file_path)
        for col in df.columns:
            if df[col].dtype == object:
                mask = df[col].astype(str).str.contains("tobacco|smoke|smoking|cigarette|alcohol", case=False, na=False)
                matches = df[mask]
                if not matches.empty:
                    print(f"Found matches in column '{col}':")
                    unique_matches = matches[col].unique()
                    for m in unique_matches[:10]:
                        print(f"  - {m}")
                    print(f"  Total unique matches in '{col}': {len(unique_matches)}")
    except Exception as e:
        print("Error:", e)
    print("\n" + "="*50 + "\n")

search_terms("global_health_estimates_life_expectancy_and_leading_causes_of_death_and_disability_indicators_lk.csv")
search_terms("air_pollution_indicators_lka.csv")
