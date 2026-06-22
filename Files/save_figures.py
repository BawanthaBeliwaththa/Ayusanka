import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="darkgrid")

print("Loading data...")
df_le_local = pd.read_csv("global_health_estimates_life_expectancy_and_leading_causes_of_death_and_disability_indicators_lk.csv")
df_air_local = pd.read_csv("air_pollution_indicators_lka.csv")

with open("who_lk_recorded_per_capita.json", "r") as f:
    alcohol_records = json.load(f)

with open("who_lk_fetched_data.json", "r") as f:
    tobacco_records_all = json.load(f)
tob_records = tobacco_records_all.get("tobacco_prevalence", [])
cig_records = tobacco_records_all.get("cig_prevalence", [])

# Preprocess
df_le = df_le_local[df_le_local['GHO (CODE)'] == 'WHOSIS_000001']
df_le_clean = df_le[['YEAR (DISPLAY)', 'DIMENSION (NAME)', 'Numeric']].rename(
    columns={'YEAR (DISPLAY)': 'Year', 'DIMENSION (NAME)': 'Sex', 'Numeric': 'LE'}
)
df_le_pivot = df_le_clean.pivot(index='Year', columns='Sex', values='LE').reset_index()
df_le_pivot.rename(columns={'Both sexes': 'LE_Both', 'Female': 'LE_Female', 'Male': 'LE_Male'}, inplace=True)

df_hale = df_le_local[df_le_local['GHO (CODE)'] == 'WHOSIS_000002']
df_hale_clean = df_hale[['YEAR (DISPLAY)', 'DIMENSION (NAME)', 'Numeric']].rename(
    columns={'YEAR (DISPLAY)': 'Year', 'DIMENSION (NAME)': 'Sex', 'Numeric': 'HALE'}
)
df_hale_pivot = df_hale_clean.pivot(index='Year', columns='Sex', values='HALE').reset_index()
df_hale_pivot.rename(columns={'Both sexes': 'HALE_Both', 'Female': 'HALE_Female', 'Male': 'HALE_Male'}, inplace=True)

df_le_hale = pd.merge(df_le_pivot, df_hale_pivot, on='Year', how='outer')

df_pm = df_air_local[(df_air_local['GHO (CODE)'] == 'SDGPM25') & (df_air_local['DIMENSION (NAME)'] == 'Total')]
df_pm_clean = df_pm[['YEAR (DISPLAY)', 'Numeric']].rename(
    columns={'YEAR (DISPLAY)': 'Year', 'Numeric': 'PM25'}
)

alc_rows = []
for r in alcohol_records:
    if r.get("Dim1") == "ALCOHOLTYPE_SA_TOTAL":
        alc_rows.append({
            "Year": r.get("TimeDim"),
            "Alcohol": r.get("NumericValue")
        })
df_alc = pd.DataFrame(alc_rows).drop_duplicates().sort_values('Year')

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

full_years = pd.DataFrame({"Year": list(range(2000, 2031))})
df_tob_full = pd.merge(full_years, df_tob_pivot, on='Year', how='left')
df_tob_full = df_tob_full.interpolate(method='linear', limit_direction='both')

df_aligned = pd.read_csv("aligned_historical_data.csv")

# 1. Generate Historical Trends Figure
print("Generating historical trends figure...")
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs[0, 0].plot(df_aligned['Year'], df_aligned['LE_Both'], marker='o', linewidth=2.5, color='#10b981', label='Both')
axs[0, 0].plot(df_aligned['Year'], df_aligned['LE_Male'], marker='s', linewidth=1.5, color='#6366f1', label='Male')
axs[0, 0].plot(df_aligned['Year'], df_aligned['LE_Female'], marker='^', linewidth=1.5, color='#ec4899', label='Female')
axs[0, 0].set_title('Life Expectancy at Birth', fontsize=12, fontweight='bold')
axs[0, 0].set_xlabel('Year')
axs[0, 0].set_ylabel('Years')
axs[0, 0].legend()
axs[0, 0].grid(True)

axs[0, 1].plot(df_aligned['Year'], df_aligned['PM25'], marker='o', linewidth=2.5, color='#06b6d4')
axs[0, 1].axhline(5.0, color='r', linestyle='--', label='WHO Target')
axs[0, 1].set_title('Air Pollution (PM2.5 Concentrations)', fontsize=12, fontweight='bold')
axs[0, 1].set_xlabel('Year')
axs[0, 1].set_ylabel('µg/m³')
axs[0, 1].legend()
axs[0, 1].grid(True)

axs[1, 0].plot(df_aligned['Year'], df_aligned['Tob_Both'], marker='o', linewidth=2.5, color='#ef4444', label='Both')
axs[1, 0].plot(df_aligned['Year'], df_aligned['Tob_Male'], marker='s', linewidth=1.5, color='#b91c1c', label='Male')
axs[1, 0].plot(df_aligned['Year'], df_aligned['Tob_Female'], marker='^', linewidth=1.5, color='#fca5a5', label='Female')
axs[1, 0].set_title('Tobacco Smoking Prevalence', fontsize=12, fontweight='bold')
axs[1, 0].set_xlabel('Year')
axs[1, 0].set_ylabel('%')
axs[1, 0].legend()
axs[1, 0].grid(True)

axs[1, 1].plot(df_aligned['Year'], df_aligned['Alcohol'], marker='o', linewidth=2.5, color='#f59e0b')
axs[1, 1].set_title('Alcohol Consumption Per Capita', fontsize=12, fontweight='bold')
axs[1, 1].set_xlabel('Year')
axs[1, 1].set_ylabel('Litres of Pure Alcohol')
axs[1, 1].grid(True)

plt.suptitle('Sri Lanka Health & Environmental Trends (2000 - 2026)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("historical_health_trends.png", dpi=300)
plt.close()
print("Saved historical_health_trends.png")

# 2. Generate Correlation Heatmap
print("Generating correlation matrix heatmap...")
columns_to_corr = ['LE_Both', 'LE_Male', 'LE_Female', 'PM25', 'Alcohol', 'Tob_Both', 'Tob_Male', 'Tob_Female']
corr_matrix = df_aligned[columns_to_corr].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=.5)
plt.title('Correlation Matrix of Sri Lankan Health Data (2000-2026)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300)
plt.close()
print("Saved correlation_heatmap.png")

# 3. Model & Forecast Scenarios
print("Calculating OLS coefficients...")
df_aligned_train = df_aligned[df_aligned['Year'] <= 2021]
coefficients = {}
for group in ['Both', 'Male', 'Female']:
    y_col = f"LE_{group}"
    tob_col = f"Tob_{group}"
    X = df_aligned_train[[tob_col, 'Alcohol', 'PM25']].values
    y = df_aligned_train[y_col].values
    X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
    beta, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
    coefficients[group] = {"intercept": beta[0], "tobacco": beta[1], "alcohol": beta[2], "pm25": beta[3]}

projection_years = list(range(2026, 2051))
base_row = df_aligned[df_aligned['Year'] == 2026].iloc[0]
base_tob = {'Both': base_row['Tob_Both'], 'Male': base_row['Tob_Male'], 'Female': base_row['Tob_Female']}
base_alc = base_row['Alcohol']
base_pm25 = base_row['PM25']

scenarios = {}
for scen_name in ['Baseline', 'Optimistic', 'Pessimistic']:
    scenarios[scen_name] = []
    for i, year in enumerate(projection_years):
        t = i
        if scen_name == 'Baseline':
            alc = max(0.5, base_alc * (1 - 0.01 * t))
            pm25 = max(5.0, base_pm25 * (1 - 0.01 * t))
            tob = {g: max(0.5, base_tob[g] * (1 - 0.015 * t)) for g in ['Both', 'Male', 'Female']}
        elif scen_name == 'Optimistic':
            alc = max(0.5, base_alc * (1 - 0.03 * t))
            pm25 = max(5.0, base_pm25 - ((base_pm25 - 5.0) / 24) * t)
            tob = {g: max(0.5, base_tob[g] * (1 - 0.03 * t)) for g in ['Both', 'Male', 'Female']}
        else:
            alc = base_alc * (1 + 0.01 * t)
            pm25 = base_pm25 * (1 + 0.015 * t)
            tob = {g: base_tob[g] * (1 + 0.005 * t) for g in ['Both', 'Male', 'Female']}
            
        predicted_le = {}
        for g in ['Both', 'Male', 'Female']:
            coef = coefficients[g]
            predicted_le[g] = (
                coef['intercept'] + 
                (coef['tobacco'] * tob[g]) + 
                (coef['alcohol'] * alc) + 
                (coef['pm25'] * pm25)
            )
            
        scenarios[scen_name].append({
            "Year": year,
            "LE_Both": predicted_le['Both']
        })

print("Generating scenario projection figure...")
plt.figure(figsize=(12, 7))
plt.plot(df_aligned['Year'], df_aligned['LE_Both'], color='black', marker='o', linewidth=3, label='Historical Actual')

for scen_name, color, style in [('Baseline', '#6366f1', '--'), ('Optimistic', '#10b981', '-'), ('Pessimistic', '#ef4444', '-.')]:
    df_scen = pd.DataFrame(scenarios[scen_name])
    connect_year = [df_aligned['Year'].iloc[-1]] + list(df_scen['Year'])
    connect_val = [df_aligned['LE_Both'].iloc[-1]] + list(df_scen['LE_Both'])
    plt.plot(connect_year, connect_val, color=color, linestyle=style, linewidth=2.5, label=f'{scen_name} Scenario')

plt.axvline(2026, color='gray', linestyle=':', label='Forecast Boundary (2026)')
plt.title('Sri Lanka Life Expectancy Scenario Projections (2026 - 2050)', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Life Expectancy (Both Sexes)')
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.grid(True)
plt.tight_layout()
plt.savefig("projection_scenarios.png", dpi=300)
plt.close()
print("Saved projection_scenarios.png")

print("All figures saved successfully!")
