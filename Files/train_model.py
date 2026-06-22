import json
import numpy as np
import pandas as pd
from db_manager import init_db, store_model_data
from export_csv import export_all

# Load data files
print("Loading datasets...")
df_le_local = pd.read_csv("global_health_estimates_life_expectancy_and_leading_causes_of_death_and_disability_indicators_lk.csv")
df_air_local = pd.read_csv("air_pollution_indicators_lka.csv")

with open("who_lk_recorded_per_capita.json", "r") as f:
    alcohol_records = json.load(f)

with open("who_lk_fetched_data.json", "r") as f:
    tobacco_records_all = json.load(f)
tob_records = tobacco_records_all.get("tobacco_prevalence", [])
cig_records = tobacco_records_all.get("cig_prevalence", [])

# 1. Process Life Expectancy at Birth (WHOSIS_000001)
print("Processing Life Expectancy...")
df_le = df_le_local[df_le_local['GHO (CODE)'] == 'WHOSIS_000001']
df_le_clean = df_le[['YEAR (DISPLAY)', 'DIMENSION (NAME)', 'Numeric']].rename(
    columns={'YEAR (DISPLAY)': 'Year', 'DIMENSION (NAME)': 'Sex', 'Numeric': 'LE'}
)
df_le_pivot = df_le_clean.pivot(index='Year', columns='Sex', values='LE').reset_index()
df_le_pivot.rename(columns={'Both sexes': 'LE_Both', 'Female': 'LE_Female', 'Male': 'LE_Male'}, inplace=True)

# Process Healthy Life Expectancy (WHOSIS_000002)
df_hale = df_le_local[df_le_local['GHO (CODE)'] == 'WHOSIS_000002']
df_hale_clean = df_hale[['YEAR (DISPLAY)', 'DIMENSION (NAME)', 'Numeric']].rename(
    columns={'YEAR (DISPLAY)': 'Year', 'DIMENSION (NAME)': 'Sex', 'Numeric': 'HALE'}
)
df_hale_pivot = df_hale_clean.pivot(index='Year', columns='Sex', values='HALE').reset_index()
df_hale_pivot.rename(columns={'Both sexes': 'HALE_Both', 'Female': 'HALE_Female', 'Male': 'HALE_Male'}, inplace=True)

df_le_hale = pd.merge(df_le_pivot, df_hale_pivot, on='Year', how='outer')

# 2. Process PM2.5 (SDGPM25)
print("Processing PM2.5 air pollution...")
df_pm = df_air_local[(df_air_local['GHO (CODE)'] == 'SDGPM25') & (df_air_local['DIMENSION (NAME)'] == 'Total')]
df_pm_clean = df_pm[['YEAR (DISPLAY)', 'Numeric']].rename(
    columns={'YEAR (DISPLAY)': 'Year', 'Numeric': 'PM25'}
)

# 3. Process Alcohol Consumption (SA_0000001400 total)
print("Processing Alcohol consumption...")
alc_rows = []
for r in alcohol_records:
    if r.get("Dim1") == "ALCOHOLTYPE_SA_TOTAL":
        alc_rows.append({
            "Year": r.get("TimeDim"),
            "Alcohol": r.get("NumericValue")
        })
df_alc = pd.DataFrame(alc_rows).drop_duplicates().sort_values('Year')

# 4. Process Tobacco Prevalence (M_Est_tob_curr)
print("Processing Tobacco prevalence...")
tob_rows = []
for r in tob_records:
    tob_rows.append({
        "Year": r.get("TimeDim"),
        "Sex": r.get("Dim1"),
        "Tobacco": r.get("NumericValue")
    })
df_tob = pd.DataFrame(tob_rows)
df_tob_pivot = df_tob.pivot(index='Year', columns='Sex', values='Tobacco').reset_index()
df_tob_pivot.rename(columns={'SEX_BTSX': 'Tob_Both', 'SEX_FMLE': 'Tob_Female', 'SEX_MLE': 'Tob_Male'}, inplace=True)

# Cigarette Prevalence (M_Est_cig_curr)
cig_rows = []
for r in cig_records:
    cig_rows.append({
        "Year": r.get("TimeDim"),
        "Sex": r.get("Dim1"),
        "Cigarette": r.get("NumericValue")
    })
df_cig = pd.DataFrame(cig_rows)
df_cig_pivot = df_cig.pivot(index='Year', columns='Sex', values='Cigarette').reset_index()
df_cig_pivot.rename(columns={'SEX_BTSX': 'Cig_Both', 'SEX_FMLE': 'Cig_Female', 'SEX_MLE': 'Cig_Male'}, inplace=True)

df_tob_cig = pd.merge(df_tob_pivot, df_cig_pivot, on='Year', how='outer')

full_years = pd.DataFrame({"Year": list(range(2000, 2031))})
df_tob_cig_full = pd.merge(full_years, df_tob_cig, on='Year', how='left')
df_tob_cig_full = df_tob_cig_full.interpolate(method='linear', limit_direction='both')

# 5. Process NCD Mortality Probability (NCDMORT3070)
print("Processing NCD mortality probability...")
df_ncd = df_le_local[df_le_local['GHO (CODE)'] == 'NCDMORT3070']
df_ncd_clean = df_ncd[['YEAR (DISPLAY)', 'DIMENSION (NAME)', 'Numeric']].rename(
    columns={'YEAR (DISPLAY)': 'Year', 'DIMENSION (NAME)': 'Sex', 'Numeric': 'NCD'}
)
df_ncd_pivot = df_ncd_clean.pivot(index='Year', columns='Sex', values='NCD').reset_index()
df_ncd_pivot.rename(columns={'Both sexes': 'NCD_Both', 'Female': 'NCD_Female', 'Male': 'NCD_Male'}, inplace=True)

# 6. Merge all datasets into one aligned DataFrame (2000 to 2026)
print("Merging datasets (2000-2026)...")
years_df = pd.DataFrame({"Year": list(range(2000, 2027))})
df_merged = pd.merge(years_df, df_le_hale, on='Year', how='left')
df_merged = pd.merge(df_merged, df_pm_clean[['Year', 'PM25']], on='Year', how='left')
df_merged = pd.merge(df_merged, df_alc, on='Year', how='left')
df_merged = pd.merge(df_merged, df_tob_cig_full, on='Year', how='left')
df_merged = pd.merge(df_merged, df_ncd_pivot, on='Year', how='left')

# Backfill PM2.5 for 2000-2009 using the 2010 value, and ffill for 2024-2026
df_merged['PM25'] = df_merged['PM25'].bfill().ffill()

# Ffill Alcohol and NCD values for missing historical years up to 2026
df_merged['Alcohol'] = df_merged['Alcohol'].ffill()
for g in ['Both', 'Male', 'Female']:
    df_merged[f'NCD_{g}'] = df_merged[f'NCD_{g}'].ffill()

# Extrapolate LE and HALE for 2022-2026 using linear trend of 2000-2021 actuals
for col in ['LE_Both', 'LE_Male', 'LE_Female', 'HALE_Both', 'HALE_Male', 'HALE_Female']:
    actual_years = df_merged[df_merged[col].notna()]['Year'].values
    actual_vals = df_merged[df_merged[col].notna()][col].values
    slope, intercept = np.polyfit(actual_years, actual_vals, 1)
    
    missing_years = df_merged[df_merged[col].isna()]['Year'].values
    for yr in missing_years:
        df_merged.loc[df_merged['Year'] == yr, col] = slope * yr + intercept

df_aligned_train = df_merged.sort_values('Year').copy()
print(f"Aligned historical training dataset covers {len(df_aligned_train)} years (2000-2026).")
print(df_aligned_train.head(27))

# 7. Fit Models (OLS and Ridge Regression) on 2000-2021 training set
# Independent variables: Tobacco, Alcohol, PM25, NCD
coefficients = {"OLS": {}, "Ridge": {}}
alpha = 0.5

for group in ['Both', 'Male', 'Female']:
    print(f"\nTraining model for: {group}")
    y_col = f"LE_{group}"
    tob_col = f"Tob_{group}"
    ncd_col = f"NCD_{group}"
    
    X = df_aligned_train[[tob_col, 'Alcohol', 'PM25', ncd_col]].values
    y = df_aligned_train[y_col].values
    
    # 7a. OLS Fit
    X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
    beta_ols, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
    y_pred_ols = np.dot(X_with_intercept, beta_ols)
    
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res_ols = np.sum((y - y_pred_ols) ** 2)
    r2_ols = 1 - (ss_res_ols / ss_tot)
    mse_ols = ss_res_ols / len(y)
    
    coefficients["OLS"][group] = {
        "intercept": float(beta_ols[0]),
        "tobacco": float(beta_ols[1]),
        "alcohol": float(beta_ols[2]),
        "pm25": float(beta_ols[3]),
        "ncd": float(beta_ols[4]),
        "r2": float(r2_ols),
        "mse": float(mse_ols)
    }
    
    # 7b. Ridge Fit (alpha=0.5)
    # Eye matrix skipping intercept regularization
    I_prime = np.eye(X_with_intercept.shape[1])
    I_prime[0, 0] = 0.0
    beta_ridge = np.linalg.inv(X_with_intercept.T @ X_with_intercept + alpha * I_prime) @ X_with_intercept.T @ y
    y_pred_ridge = np.dot(X_with_intercept, beta_ridge)
    
    ss_res_ridge = np.sum((y - y_pred_ridge) ** 2)
    r2_ridge = 1 - (ss_res_ridge / ss_tot)
    mse_ridge = ss_res_ridge / len(y)
    
    coefficients["Ridge"][group] = {
        "intercept": float(beta_ridge[0]),
        "tobacco": float(beta_ridge[1]),
        "alcohol": float(beta_ridge[2]),
        "pm25": float(beta_ridge[3]),
        "ncd": float(beta_ridge[4]),
        "r2": float(r2_ridge),
        "mse": float(mse_ridge)
    }
    
    print(f"  OLS   Formula: LE = {beta_ols[0]:.4f} + ({beta_ols[1]:.4f} * Tob) + ({beta_ols[2]:.4f} * Alc) + ({beta_ols[3]:.4f} * PM2.5) + ({beta_ols[4]:.4f} * NCD)")
    print(f"  OLS   R2: {r2_ols:.4f}, MSE: {mse_ols:.4f}")
    print(f"  Ridge Formula: LE = {beta_ridge[0]:.4f} + ({beta_ridge[1]:.4f} * Tob) + ({beta_ridge[2]:.4f} * Alc) + ({beta_ridge[3]:.4f} * PM2.5) + ({beta_ridge[4]:.4f} * NCD)")
    print(f"  Ridge R2: {r2_ridge:.4f}, MSE: {mse_ridge:.4f}")

# 8. Use aligned training data as historical aligned data
df_aligned = df_aligned_train.copy()
print("Full historical dataset covers years:")
print(df_aligned['Year'].values)

df_aligned.to_csv("aligned_historical_data.csv", index=False)


# 9. Generate Projections for 2026 to 2050 (Year-Wise)
print("\nGenerating 25-Year Projections (2026-2050)...")
projection_years = list(range(2026, 2051))

base_row = df_aligned[df_aligned['Year'] == 2026].iloc[0]
base_tob = {'Both': base_row['Tob_Both'], 'Male': base_row['Tob_Male'], 'Female': base_row['Tob_Female']}
base_cig = {'Both': base_row['Cig_Both'], 'Male': base_row['Cig_Male'], 'Female': base_row['Cig_Female']}
base_ncd = {'Both': base_row['NCD_Both'], 'Male': base_row['NCD_Male'], 'Female': base_row['NCD_Female']}
base_alc = base_row['Alcohol']
base_pm25 = base_row['PM25']

scenarios = {}

for scen_name in ['Baseline', 'Optimistic', 'Pessimistic']:
    scenarios[scen_name] = []
    
    for i, year in enumerate(projection_years):
        t = i  # Years since 2026 (first year is 2026 itself, i.e., t=0)
        
        if scen_name == 'Baseline':
            alc = max(0.5, base_alc * (1 - 0.01 * t))
            pm25 = max(5.0, base_pm25 * (1 - 0.01 * t))
            tob = {g: max(0.5, base_tob[g] * (1 - 0.015 * t)) for g in ['Both', 'Male', 'Female']}
            cig = {g: max(0.1, base_cig[g] * (1 - 0.015 * t)) for g in ['Both', 'Male', 'Female']}
            ncd = {g: max(5.0, base_ncd[g] * (1 - 0.015 * t)) for g in ['Both', 'Male', 'Female']}
            
        elif scen_name == 'Optimistic':
            alc = max(0.5, base_alc * (1 - 0.03 * t))
            pm25 = max(5.0, base_pm25 - ((base_pm25 - 5.0) / 24) * t)
            tob = {g: max(0.5, base_tob[g] * (1 - 0.03 * t)) for g in ['Both', 'Male', 'Female']}
            cig = {g: max(0.1, base_cig[g] * (1 - 0.03 * t)) for g in ['Both', 'Male', 'Female']}
            ncd = {g: max(5.0, base_ncd[g] * (1 - 0.03 * t)) for g in ['Both', 'Male', 'Female']}
            
        else:  # Pessimistic
            alc = base_alc * (1 + 0.01 * t)
            pm25 = base_pm25 * (1 + 0.015 * t)
            tob = {g: base_tob[g] * (1 + 0.005 * t) for g in ['Both', 'Male', 'Female']}
            cig = {g: base_cig[g] * (1 + 0.005 * t) for g in ['Both', 'Male', 'Female']}
            ncd = {g: base_ncd[g] * (1 + 0.005 * t) for g in ['Both', 'Male', 'Female']}
            
        # Predict Life Expectancy for both OLS and Ridge models
        pred_ols = {}
        pred_ridge = {}
        for g in ['Both', 'Male', 'Female']:
            c_ols = coefficients["OLS"][g]
            pred_ols[g] = (
                c_ols['intercept'] + 
                (c_ols['tobacco'] * tob[g]) + 
                (c_ols['alcohol'] * alc) + 
                (c_ols['pm25'] * pm25) + 
                (c_ols['ncd'] * ncd[g])
            )
            
            c_rdg = coefficients["Ridge"][g]
            pred_ridge[g] = (
                c_rdg['intercept'] + 
                (c_rdg['tobacco'] * tob[g]) + 
                (c_rdg['alcohol'] * alc) + 
                (c_rdg['pm25'] * pm25) + 
                (c_rdg['ncd'] * ncd[g])
            )
            
        scenarios[scen_name].append({
            "Year": year,
            "Alcohol": float(alc),
            "PM25": float(pm25),
            "Tob_Both": float(tob['Both']),
            "Tob_Male": float(tob['Male']),
            "Tob_Female": float(tob['Female']),
            "Cig_Both": float(cig['Both']),
            "Cig_Male": float(cig['Male']),
            "Cig_Female": float(cig['Female']),
            "NCD_Both": float(ncd['Both']),
            "NCD_Male": float(ncd['Male']),
            "NCD_Female": float(ncd['Female']),
            # Default to Ridge for backward compatibility
            "LE_Both": float(pred_ridge['Both']),
            "LE_Male": float(pred_ridge['Male']),
            "LE_Female": float(pred_ridge['Female']),
            # Specific values
            "LE_OLS_Both": float(pred_ols['Both']),
            "LE_OLS_Male": float(pred_ols['Male']),
            "LE_OLS_Female": float(pred_ols['Female']),
            "LE_Ridge_Both": float(pred_ridge['Both']),
            "LE_Ridge_Male": float(pred_ridge['Male']),
            "LE_Ridge_Female": float(pred_ridge['Female'])
        })

# 9. Export to model_data.json
print("\nExporting model parameters and data to model_data.json...")

historical_records = []
for idx, r in df_aligned.iterrows():
    historical_records.append({
        "Year": int(r['Year']),
        "LE_Both": float(r['LE_Both']),
        "LE_Male": float(r['LE_Male']),
        "LE_Female": float(r['LE_Female']),
        "HALE_Both": float(r['HALE_Both']),
        "HALE_Male": float(r['HALE_Male']),
        "HALE_Female": float(r['HALE_Female']),
        "PM25": float(r['PM25']),
        "Alcohol": float(r['Alcohol']),
        "Tob_Both": float(r['Tob_Both']),
        "Tob_Male": float(r['Tob_Male']),
        "Tob_Female": float(r['Tob_Female']),
        "Cig_Both": float(r['Cig_Both']),
        "Cig_Male": float(r['Cig_Male']),
        "Cig_Female": float(r['Cig_Female']),
        "NCD_Both": float(r['NCD_Both']),
        "NCD_Male": float(r['NCD_Male']),
        "NCD_Female": float(r['NCD_Female'])
    })

# Districts PM2.5 mapping for individual prediction
districts_pm25 = [
    {"district": "Colombo", "pm25": 22.7, "zone": "Western"},
    {"district": "Gampaha", "pm25": 21.0, "zone": "Western"},
    {"district": "Kalutara", "pm25": 18.0, "zone": "Western"},
    {"district": "Kandy", "pm25": 19.5, "zone": "Central"},
    {"district": "Matale", "pm25": 16.5, "zone": "Central"},
    {"district": "Nuwara Eliya", "pm25": 8.5, "zone": "Central"},
    {"district": "Galle", "pm25": 14.0, "zone": "Southern"},
    {"district": "Matara", "pm25": 13.5, "zone": "Southern"},
    {"district": "Hambantota", "pm25": 14.0, "zone": "Southern"},
    {"district": "Jaffna", "pm25": 15.5, "zone": "Northern"},
    {"district": "Mannar", "pm25": 13.0, "zone": "Northern"},
    {"district": "Vavuniya", "pm25": 14.0, "zone": "Northern"},
    {"district": "Mullaitivu", "pm25": 12.0, "zone": "Northern"},
    {"district": "Kilinochchi", "pm25": 13.5, "zone": "Northern"},
    {"district": "Batticaloa", "pm25": 12.5, "zone": "Eastern"},
    {"district": "Ampara", "pm25": 13.0, "zone": "Eastern"},
    {"district": "Trincomalee", "pm25": 13.0, "zone": "Eastern"},
    {"district": "Kurunegala", "pm25": 17.5, "zone": "North Western"},
    {"district": "Puttalam", "pm25": 16.5, "zone": "North Western"},
    {"district": "Anuradhapura", "pm25": 16.0, "zone": "North Central"},
    {"district": "Polonnaruwa", "pm25": 15.0, "zone": "North Central"},
    {"district": "Badulla", "pm25": 12.0, "zone": "Uva"},
    {"district": "Moneragala", "pm25": 14.0, "zone": "Uva"},
    {"district": "Ratnapura", "pm25": 18.5, "zone": "Sabaragamuwa"},
    {"district": "Kegalle", "pm25": 16.0, "zone": "Sabaragamuwa"}
]

output_db = {
    "metadata": {
        "title": "AyuSanka Model Database v2",
        "description": "Sri Lankan Life Expectancy macro parameters (OLS & Ridge), historical dataset, and scenario projections (2026-2050)",
        "source": "WHO Global Health Observatory & Sri Lanka MOH Aligned Datasets",
        "lastUpdated": "2026-06-10"
    },
    "coefficients": coefficients,
    "historical": historical_records,
    "projections": scenarios,
    "districts": districts_pm25
}

with open("model_data.json", "w") as f:
    json.dump(output_db, f, indent=4)

# Copy to frontend src if frontend path exists
try:
    with open("frontend/src/model_data.json", "w") as f:
        json.dump(output_db, f, indent=4)
    print("model_data.json copied to frontend/src/ successfully!")
except Exception as e:
    print(f"Could not copy to frontend/src/: {e}")

print("model_data.json written successfully!")

# 10. Store all data to TinyDB NoSQL Database
print("\n" + "="*60)
print("STORING DATA TO NoSQL DATABASE (TinyDB)")
print("="*60)
db = init_db()
store_model_data(db, coefficients, historical_records, scenarios, districts_pm25)

# 11. Export all data to CSV files
print("="*60)
print("EXPORTING ALL DATA TO CSV")
print("="*60)
export_all(
    db=db,
    coefficients_dict=coefficients,
    historical_records=historical_records,
    scenarios=scenarios,
    districts_list=districts_pm25
)

db.close()
print("\nAll pipeline steps complete: JSON + NoSQL DB + CSV exports.")
