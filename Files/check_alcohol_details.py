import json
import pandas as pd

with open("who_lk_recorded_per_capita.json", "r") as f:
    records = json.load(f)

print(f"Total records found for SA_0000001400: {len(records)}")

rows = []
for r in records:
    rows.append({
        "Year": r.get("TimeDim"),
        "Dim1": r.get("Dim1"),
        "Dim2": r.get("Dim2"),
        "Value": r.get("NumericValue"),
        "ValueStr": r.get("Value")
    })
df = pd.DataFrame(rows)
print("\nUnique Dim1 values (usually beverage types):")
print(df['Dim1'].value_counts())
print("\nUnique Dim2 values (usually sex):")
print(df['Dim2'].value_counts())

print("\nSample records for a recent year (e.g., 2019):")
print(df[df['Year'] == 2019])
