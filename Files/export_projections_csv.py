import json
import pandas as pd

def export_all():
    print("Loading model_data.json...")
    with open("model_data.json", "r") as f:
        db = json.load(f)
        
    print("Exporting Historical Aligned Data...")
    df_hist = pd.DataFrame(db['historical'])
    df_hist.to_csv("aligned_historical_data.csv", index=False)
    print("Saved: aligned_historical_data.csv")
    
    projections = db['projections']
    for scenario_name, records in projections.items():
        filename = f"projected_{scenario_name.lower()}_2026_2050.csv"
        print(f"Exporting {scenario_name} Projection...")
        df_proj = pd.DataFrame(records)
        df_proj.to_csv(filename, index=False)
        print(f"Saved: {filename}")
        
    print("All datasets exported successfully!")

if __name__ == "__main__":
    export_all()
