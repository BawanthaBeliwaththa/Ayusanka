import json
import requests
import pandas as pd

with open("who_indicators.json", "r") as f:
    indicators_meta = json.load(f)

alcohol_inds = indicators_meta.get("alcohol", [])
print(f"Total alcohol indicators in metadata: {len(alcohol_inds)}")

# Let's search for indicators related to per capita consumption (litres)
litres_indicators = [ind for ind in alcohol_inds if "litres" in ind[1].lower() or "per capita" in ind[1].lower()]
print(f"\nIndicators with 'litres' or 'per capita' ({len(litres_indicators)}):")
for code, name in litres_indicators[:15]:
    print(f"- {code}: {name}")

# Let's try to query some of them for Sri Lanka
test_codes = {
    "recorded_per_capita": "SA_0000001400",
    "recorded_3yr_avg": "SA_0000001401",
    "total_per_capita": "SA_0000001403",
    "total_per_capita_alternative": "ALCOHOL_PERCAPITA" # common WHO code
}

# Let's fetch data for each of these for Sri Lanka
for name, code in test_codes.items():
    print(f"\nFetching {name} ({code})...")
    url = f"https://ghoapi.azureedge.net/api/{code}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json().get("value", [])
            lk_records = [r for r in data if r.get("SpatialDim") == "LKA" or r.get("SpatialDimValueCode") == "LKA"]
            print(f"  Found {len(lk_records)} records for Sri Lanka.")
            if lk_records:
                years = sorted(list(set(r.get("TimeDim") for r in lk_records)))
                print(f"  Years: {years}")
                # Save it if we have records
                with open(f"who_lk_{name}.json", "w") as f:
                    json.dump(lk_records, f, indent=4)
        else:
            print(f"  Failed. Status code: {response.status_code}")
    except Exception as e:
        print(f"  Error: {e}")
