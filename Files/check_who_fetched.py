import json
import pandas as pd

with open("who_lk_fetched_data.json", "r") as f:
    data = json.load(f)

for indicator_name, records in data.items():
    print(f"\n=== {indicator_name} (Total: {len(records)}) ===")
    if not records:
        print("No records found.")
        continue
    
    rows = []
    for r in records:
        rows.append({
            "Year": r.get("TimeDim"),
            "Sex": r.get("Dim1"),
            "Value": r.get("NumericValue"),
            "StringValue": r.get("Value")
        })
    df = pd.DataFrame(rows)
    # Pivot for easier reading
    df_pivoted = df.pivot_table(index="Year", columns="Sex", values="Value", aggfunc='first')
    print(df_pivoted.sort_index(ascending=False))
