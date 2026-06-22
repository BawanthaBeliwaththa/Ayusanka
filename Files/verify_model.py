import json
import os

def verify():
    print("=== Verifying Model Database ===")
    
    file_path = "model_data.json"
    if not os.path.exists(file_path):
        print("FAIL: model_data.json does not exist!")
        return
        
    try:
        with open(file_path, "r") as f:
            db = json.load(f)
            
        print("1. Schema Structure Check:")
        required_keys = ["metadata", "coefficients", "historical", "projections", "districts"]
        for key in required_keys:
            if key in db:
                print(f"  - Key '{key}' found. [OK]")
            else:
                print(f"  - Key '{key}' NOT found! [FAIL]")
                return
                
        print("\n2. Coefficients Validation:")
        for model in ['OLS', 'Ridge']:
            print(f"  * Model Type: {model}")
            for group in ['Both', 'Male', 'Female']:
                coefs = db['coefficients'].get(model, {}).get(group)
                if not coefs:
                    print(f"    - Missing coefficients for {model} -> {group}! [FAIL]")
                    return
                print(f"    - {group} model: R2={coefs['r2']:.4f}, MSE={coefs['mse']:.4f}, Intercept={coefs['intercept']:.4f}, Tob={coefs['tobacco']:.4f}, Alc={coefs['alcohol']:.4f}, PM25={coefs['pm25']:.4f}, NCD={coefs['ncd']:.4f}")
                if coefs['r2'] <= 0 or coefs['r2'] > 1.0:
                    print(f"      WARNING: Unrealistic R-squared: {coefs['r2']}")
                
        print("\n3. Historical Records Validation:")
        hist = db['historical']
        print(f"  - Found {len(hist)} historical aligned records.")
        if len(hist) < 10:
            print("    FAIL: Less than 10 years of historical data!")
            return
        print(f"  - Year range: {hist[0]['Year']} to {hist[-1]['Year']}")
        
        print("\n4. Projections Validation:")
        proj = db['projections']
        for scen in ['Baseline', 'Optimistic', 'Pessimistic']:
            scen_data = proj.get(scen, [])
            print(f"  - Scenario '{scen}' has {len(scen_data)} projected years.")
            if len(scen_data) != 25:
                print(f"    FAIL: Scenario '{scen}' should have exactly 25 years (2026-2050)!")
                return
                
        print("\n5. Districts Validation:")
        dist = db['districts']
        print(f"  - Found {len(dist)} districts mapped for individual calculations.")
        if len(dist) != 25:
            print("    FAIL: Expected 25 districts in Sri Lanka!")
            return
            
        print("\nVERIFICATION SUCCESSFUL: Model database is valid and ready for UI integration!")
        
    except Exception as e:
        print("FAIL: An error occurred during validation:", e)

if __name__ == "__main__":
    verify()
