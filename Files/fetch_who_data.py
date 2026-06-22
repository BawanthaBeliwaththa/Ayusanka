import requests
import json

indicators_to_fetch = {
    "tobacco_prevalence": "M_Est_tob_curr",
    "tobacco_daily": "M_Est_tob_daily",
    "cig_prevalence": "M_Est_cig_curr",
    "alcohol_consumption": "SA_0000001403"
}

results = {}

for name, code in indicators_to_fetch.items():
    print(f"Fetching {name} ({code})...")
    url = f"https://ghoapi.azureedge.net/api/{code}"
    try:
        # We fetch all data and filter locally for LKA to be safe with OData filter syntax
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            all_records = data.get("value", [])
            lk_records = [r for r in all_records if r.get("SpatialDim") == "LKA" or r.get("SpatialDimValueCode") == "LKA"]
            results[name] = lk_records
            print(f"  Found {len(lk_records)} records for Sri Lanka.")
            if lk_records:
                print("  Sample record:", json.dumps(lk_records[0], indent=2))
        else:
            print(f"  Failed. Status code: {response.status_code}")
    except Exception as e:
        print("  Error:", e)

with open("who_lk_fetched_data.json", "w") as f:
    json.dump(results, f, indent=4)
