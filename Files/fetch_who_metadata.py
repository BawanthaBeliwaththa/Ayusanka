import requests
import json

api_url = "https://ghoapi.azureedge.net/api/Indicator"
print("Fetching indicators from WHO GHO API...")
try:
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        indicators = data.get("value", [])
        print(f"Total indicators found: {len(indicators)}")
        
        tobacco_indicators = []
        alcohol_indicators = []
        
        for ind in indicators:
            code = ind.get("IndicatorCode", "")
            name = ind.get("IndicatorName", "")
            
            if any(term in name.lower() for term in ["tobacco", "smoking", "smoke", "cigarette"]):
                tobacco_indicators.append((code, name))
            if any(term in name.lower() for term in ["alcohol", "drinking"]):
                alcohol_indicators.append((code, name))
                
        print(f"\n--- Tobacco Indicators ({len(tobacco_indicators)}) ---")
        for code, name in tobacco_indicators[:20]:
            print(f"{code}: {name}")
            
        print(f"\n--- Alcohol Indicators ({len(alcohol_indicators)}) ---")
        for code, name in alcohol_indicators[:20]:
            print(f"{code}: {name}")
            
        # Save search results
        with open("who_indicators.json", "w") as f:
            json.dump({"tobacco": tobacco_indicators, "alcohol": alcohol_indicators}, f, indent=4)
            
    else:
        print(f"Failed to fetch. Status code: {response.status_code}")
except Exception as e:
    print("Error:", e)
