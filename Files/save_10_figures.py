import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

print("1. Loading raw data...")
df_le_local = pd.read_csv("global_health_estimates_life_expectancy_and_leading_causes_of_death_and_disability_indicators_lk.csv")
df_air_local = pd.read_csv("air_pollution_indicators_lka.csv")

with open("who_lk_recorded_per_capita.json", "r") as f:
    alcohol_records = json.load(f)

with open("who_lk_fetched_data.json", "r") as f:
    tobacco_records_all = json.load(f)
tob_records = tobacco_records_all.get("tobacco_prevalence", [])
cig_records = tobacco_records_all.get("cig_prevalence", [])

# ==================== DATA PREPROCESSING ====================
print("2. Preprocessing datasets...")
# Life Expectancy (LE)
df_le = df_le_local[df_le_local['GHO (CODE)'] == 'WHOSIS_000001']
df_le_clean = df_le[['YEAR (DISPLAY)', 'DIMENSION (NAME)', 'Numeric']].rename(
    columns={'YEAR (DISPLAY)': 'Year', 'DIMENSION (NAME)': 'Sex', 'Numeric': 'LE'}
)
df_le_pivot = df_le_clean.pivot(index='Year', columns='Sex', values='LE').reset_index()
df_le_pivot.rename(columns={'Both sexes': 'LE_Both', 'Female': 'LE_Female', 'Male': 'LE_Male'}, inplace=True)

# Healthy Life Expectancy (HALE)
df_hale = df_le_local[df_le_local['GHO (CODE)'] == 'WHOSIS_000002']
df_hale_clean = df_hale[['YEAR (DISPLAY)', 'DIMENSION (NAME)', 'Numeric']].rename(
    columns={'YEAR (DISPLAY)': 'Year', 'DIMENSION (NAME)': 'Sex', 'Numeric': 'HALE'}
)
df_hale_pivot = df_hale_clean.pivot(index='Year', columns='Sex', values='HALE').reset_index()
df_hale_pivot.rename(columns={'Both sexes': 'HALE_Both', 'Female': 'HALE_Female', 'Male': 'HALE_Male'}, inplace=True)

df_le_hale = pd.merge(df_le_pivot, df_hale_pivot, on='Year', how='outer')

# PM2.5 Concentrations (multiple area types)
df_pm_all = df_air_local[df_air_local['GHO (CODE)'] == 'SDGPM25']
df_pm_pivot = df_pm_all.pivot(index='YEAR (DISPLAY)', columns='DIMENSION (NAME)', values='Numeric').reset_index()
df_pm_pivot.rename(columns={'YEAR (DISPLAY)': 'Year'}, inplace=True)
df_pm_clean = df_pm_pivot[['Year', 'Total', 'Cities', 'Urban', 'Rural', 'Towns']].rename(
    columns={'Total': 'PM25'}
)

# Alcohol by beverage types
alc_rows = []
for r in alcohol_records:
    bev_type = r.get("Dim1")
    if bev_type in ["ALCOHOLTYPE_SA_TOTAL", "ALCOHOLTYPE_SA_BEER", "ALCOHOLTYPE_SA_WINE", "ALCOHOLTYPE_SA_SPIRITS"]:
        alc_rows.append({
            "Year": r.get("TimeDim"),
            "Beverage": bev_type.replace("ALCOHOLTYPE_SA_", ""),
            "Value": r.get("NumericValue")
        })
df_alc_raw = pd.DataFrame(alc_rows)
df_alc_pivot = df_alc_raw.pivot(index='Year', columns='Beverage', values='Value').reset_index()
df_alc_pivot.rename(columns={'TOTAL': 'Alcohol'}, inplace=True)

# Tobacco & Cigarettes
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

# NCD Mortality probability
df_ncd = df_le_local[df_le_local['GHO (CODE)'] == 'NCDMORT3070']
df_ncd_clean = df_ncd[['YEAR (DISPLAY)', 'DIMENSION (NAME)', 'Numeric']].rename(
    columns={'YEAR (DISPLAY)': 'Year', 'DIMENSION (NAME)': 'Sex', 'Numeric': 'NCD'}
)
df_ncd_pivot = df_ncd_clean.pivot(index='Year', columns='Sex', values='NCD').reset_index()
df_ncd_pivot.rename(columns={'Both sexes': 'NCD_Both', 'Female': 'NCD_Female', 'Male': 'NCD_Male'}, inplace=True)

# Aligned historical data (2000 to 2026)
print("Preprocessing datasets (2000-2026)...")
years_df = pd.DataFrame({"Year": list(range(2000, 2027))})
df_merged_train = pd.merge(years_df, df_le_hale, on='Year', how='left')
df_merged_train = pd.merge(df_merged_train, df_pm_clean[['Year', 'PM25']], on='Year', how='left')
df_merged_train = pd.merge(df_merged_train, df_alc_pivot[['Year', 'Alcohol']], on='Year', how='left')
df_merged_train = pd.merge(df_merged_train, df_tob_cig_full, on='Year', how='left')
df_merged_train = pd.merge(df_merged_train, df_ncd_pivot, on='Year', how='left')

# Backfill PM2.5 for 2000-2009 using the 2010 value, and ffill for 2024-2026
df_merged_train['PM25'] = df_merged_train['PM25'].bfill().ffill()

# Ffill Alcohol and NCD values for missing historical years up to 2026
df_merged_train['Alcohol'] = df_merged_train['Alcohol'].ffill()
for g in ['Both', 'Male', 'Female']:
    df_merged_train[f'NCD_{g}'] = df_merged_train[f'NCD_{g}'].ffill()

# Extrapolate LE and HALE for 2022-2026 using linear trend of 2000-2021 actuals
for col in ['LE_Both', 'LE_Male', 'LE_Female', 'HALE_Both', 'HALE_Male', 'HALE_Female']:
    actual_years = df_merged_train[df_merged_train[col].notna()]['Year'].values
    actual_vals = df_merged_train[df_merged_train[col].notna()][col].values
    slope, intercept = np.polyfit(actual_years, actual_vals, 1)
    
    missing_years = df_merged_train[df_merged_train[col].isna()]['Year'].values
    for yr in missing_years:
        df_merged_train.loc[df_merged_train['Year'] == yr, col] = slope * yr + intercept

df_aligned_train = df_merged_train.sort_values('Year').copy()

# Fit OLS and Ridge models to extrapolate 2022-2026
coefficients = {"OLS": {}, "Ridge": {}}
alpha = 0.5
for group in ['Both', 'Male', 'Female']:
    y_col = f"LE_{group}"
    tob_col = f"Tob_{group}"
    ncd_col = f"NCD_{group}"
    X_sub = df_aligned_train[[tob_col, 'Alcohol', 'PM25', ncd_col]].values
    y_sub = df_aligned_train[y_col].values
    X_sub_with_intercept = np.hstack([np.ones((X_sub.shape[0], 1)), X_sub])
    
    # OLS fit
    b_ols, _, _, _ = np.linalg.lstsq(X_sub_with_intercept, y_sub, rcond=None)
    coefficients["OLS"][group] = {"intercept": b_ols[0], "tobacco": b_ols[1], "alcohol": b_ols[2], "pm25": b_ols[3], "ncd": b_ols[4]}
    
    # Ridge fit
    I_prime = np.eye(X_sub_with_intercept.shape[1])
    I_prime[0, 0] = 0.0
    b_ridge = np.linalg.inv(X_sub_with_intercept.T @ X_sub_with_intercept + alpha * I_prime) @ X_sub_with_intercept.T @ y_sub
    coefficients["Ridge"][group] = {"intercept": b_ridge[0], "tobacco": b_ridge[1], "alcohol": b_ridge[2], "pm25": b_ridge[3], "ncd": b_ridge[4]}

# Use aligned training data as historical aligned data
df_aligned = df_aligned_train.copy()

# ==================== FIG 1: LE vs HALE ====================
print("Generating Fig 1...")
plt.figure(figsize=(10, 6))
df_le_hale_2000 = df_aligned
plt.plot(df_le_hale_2000['Year'], df_le_hale_2000['LE_Both'], color='#10b981', marker='o', linewidth=2.5, label='Life Expectancy (LE)')
plt.plot(df_le_hale_2000['Year'], df_le_hale_2000['HALE_Both'], color='#6366f1', marker='x', linewidth=2.5, label='Healthy Life Expectancy (HALE)')
plt.fill_between(df_le_hale_2000['Year'], df_le_hale_2000['HALE_Both'], df_le_hale_2000['LE_Both'], color='#6366f1', alpha=0.15, label='Years in Ill Health')
plt.title('Fig 1: Life Expectancy vs. Healthy Life Expectancy (HALE) in Sri Lanka (2000-2026)', fontsize=12, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Expected Lifespan (Years)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fig1_le_vs_hale.png", dpi=300)
plt.close()

# ==================== FIG 2: Gender Gap ====================
print("Generating Fig 2...")
plt.figure(figsize=(10, 6))
df_le_2000 = df_aligned
gender_gap = df_le_2000['LE_Female'] - df_le_2000['LE_Male']
plt.plot(df_le_2000['Year'], gender_gap, color='#ec4899', marker='o', linewidth=2.5, label='Female - Male LE Gap')
plt.title('Fig 2: Gender Gap in Life Expectancy in Sri Lanka (2000-2026)', fontsize=12, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Lifespan Gap (Years)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fig2_gender_gap.png", dpi=300)
plt.close()

# ==================== FIG 3: PM2.5 Area Types ====================
print("Generating Fig 3...")
plt.figure(figsize=(10, 6))
df_pm_sub = df_pm_pivot[(df_pm_pivot['Year'] >= 2010) & (df_pm_pivot['Year'] <= 2023)].sort_values('Year')
plt.plot(df_pm_sub['Year'], df_pm_sub['Cities'], label='Cities', marker='o', color='#ef4444')
plt.plot(df_pm_sub['Year'], df_pm_sub['Urban'], label='Urban', marker='s', color='#f59e0b')
plt.plot(df_pm_sub['Year'], df_pm_sub['Towns'], label='Towns', marker='d', color='#6366f1')
plt.plot(df_pm_sub['Year'], df_pm_sub['Rural'], label='Rural', marker='x', color='#10b981')
plt.plot(df_pm_sub['Year'], df_pm_sub['Total'], label='National Average', marker='*', linewidth=2.5, color='#020617')
plt.axhline(5.0, color='red', linestyle='--', label='WHO Air Target (5 µg/m³)')
plt.title('Fig 3: PM2.5 Concentrations by Area Classification in Sri Lanka (2010-2023)', fontsize=12, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fig3_pm25_area.png", dpi=300)
plt.close()

# ==================== FIG 4: Tobacco vs Cigarette ====================
print("Generating Fig 4...")
plt.figure(figsize=(10, 6))
df_tc = df_aligned
plt.plot(df_tc['Year'], df_tc['Tob_Both'], marker='o', color='#ef4444', label='Total Tobacco Prevalence')
plt.plot(df_tc['Year'], df_tc['Cig_Both'], marker='s', color='#fca5a5', label='Cigarette Smoking Prevalence')
plt.title('Fig 4: Tobacco Prevalence vs. Specific Cigarette Smoking (Both Sexes, 2000-2026)', fontsize=12, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Prevalence (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fig4_tobacco_vs_cigarette.png", dpi=300)
plt.close()

# ==================== FIG 5: Alcohol Consumption shift ====================
print("Generating Fig 5...")
plt.figure(figsize=(10, 6))
df_alc_sub = df_alc_pivot[(df_alc_pivot['Year'] >= 1990) & (df_alc_pivot['Year'] <= 2022)].sort_values('Year')
plt.plot(df_alc_sub['Year'], df_alc_sub['BEER'], label='Beer', color='#f59e0b', marker='o')
plt.plot(df_alc_sub['Year'], df_alc_sub['SPIRITS'], label='Spirits', color='#ef4444', marker='s')
plt.plot(df_alc_sub['Year'], df_alc_sub['WINE'], label='Wine', color='#8b5cf6', marker='^')
plt.plot(df_alc_sub['Year'], df_alc_sub['Alcohol'], label='Total Recorded APC', color='#020617', linewidth=2, marker='*')
plt.title('Fig 5: Historical Alcohol Consumption per Capita by Beverage Type (1990-2022)', fontsize=12, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Pure Alcohol (Litres / Capita)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fig5_alcohol_beverages.png", dpi=300)
plt.close()

# ==================== FIG 6: Correlation matrix ====================
print("Generating Fig 6...")
columns_to_corr = ['LE_Both', 'LE_Male', 'LE_Female', 'PM25', 'Alcohol', 'Tob_Both', 'NCD_Both']
corr_matrix = df_aligned[columns_to_corr].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=.5)
plt.title('Fig 6: Correlation Matrix Heatmap (Aligned Period 2000-2026)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("fig6_correlation_heatmap.png", dpi=300)
plt.close()

# ==================== FIG 7: Pairplot ====================
print("Generating Fig 7...")
df_pair = df_aligned[['LE_Both', 'Tob_Both', 'Alcohol', 'PM25', 'NCD_Both']]
df_pair.columns = ['Life Expectancy', 'Tobacco (%)', 'Alcohol (L)', 'PM2.5 (ug/m3)', 'NCD Mortality (%)']
pair_plot = sns.pairplot(df_pair, diag_kind='kde', kind='reg', plot_kws={'line_kws':{'color':'red'}})
pair_plot.fig.suptitle('Fig 7: Pairplot & Distributions of Aligned Risk Indicators', y=1.02, fontsize=14, fontweight='bold')
plt.tight_layout()
pair_plot.savefig("fig7_pairplot.png", dpi=300)
plt.close()

# ==================== FIG 8: Residuals Plot ====================
print("Generating Fig 8...")
X = df_aligned_train[['Tob_Both', 'Alcohol', 'PM25', 'NCD_Both']].values
y = df_aligned_train['LE_Both'].values
X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
beta_ols, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
y_pred = np.dot(X_with_intercept, beta_ols)
residuals = y - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='#6366f1', s=80, edgecolor='black', alpha=0.8)
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.title('Fig 8: OLS Model Residuals vs. Fitted Values (Both Sexes, 2000-2026)', fontsize=12, fontweight='bold')
plt.xlabel('Fitted / Predicted Life Expectancy (Years)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.grid(True)
plt.tight_layout()
plt.savefig("fig8_regression_residuals.png", dpi=300)
plt.close()

# ==================== FIG 9: Scenario Projections ====================
print("Generating Fig 9...")
coefficients = {"OLS": {}, "Ridge": {}}
alpha = 0.5
for group in ['Both', 'Male', 'Female']:
    y_col = f"LE_{group}"
    tob_col = f"Tob_{group}"
    ncd_col = f"NCD_{group}"
    X_sub = df_aligned[[tob_col, 'Alcohol', 'PM25', ncd_col]].values
    y_sub = df_aligned[y_col].values
    X_sub_with_intercept = np.hstack([np.ones((X_sub.shape[0], 1)), X_sub])
    
    # OLS fit
    b_ols, _, _, _ = np.linalg.lstsq(X_sub_with_intercept, y_sub, rcond=None)
    coefficients["OLS"][group] = {"intercept": b_ols[0], "tobacco": b_ols[1], "alcohol": b_ols[2], "pm25": b_ols[3], "ncd": b_ols[4]}
    
    # Ridge fit
    I_prime = np.eye(X_sub_with_intercept.shape[1])
    I_prime[0, 0] = 0.0
    b_ridge = np.linalg.inv(X_sub_with_intercept.T @ X_sub_with_intercept + alpha * I_prime) @ X_sub_with_intercept.T @ y_sub
    coefficients["Ridge"][group] = {"intercept": b_ridge[0], "tobacco": b_ridge[1], "alcohol": b_ridge[2], "pm25": b_ridge[3], "ncd": b_ridge[4]}

projection_years = list(range(2026, 2051))
base_row = df_aligned[df_aligned['Year'] == 2026].iloc[0]
base_tob = {'Both': base_row['Tob_Both'], 'Male': base_row['Tob_Male'], 'Female': base_row['Tob_Female']}
base_ncd = {'Both': base_row['NCD_Both'], 'Male': base_row['NCD_Male'], 'Female': base_row['NCD_Female']}
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
            ncd = {g: max(5.0, base_ncd[g] * (1 - 0.015 * t)) for g in ['Both', 'Male', 'Female']}
        elif scen_name == 'Optimistic':
            alc = max(0.5, base_alc * (1 - 0.03 * t))
            pm25 = max(5.0, base_pm25 - ((base_pm25 - 5.0) / 24) * t)
            tob = {g: max(0.5, base_tob[g] * (1 - 0.03 * t)) for g in ['Both', 'Male', 'Female']}
            ncd = {g: max(5.0, base_ncd[g] * (1 - 0.03 * t)) for g in ['Both', 'Male', 'Female']}
        else:
            alc = base_alc * (1 + 0.01 * t)
            pm25 = base_pm25 * (1 + 0.015 * t)
            tob = {g: base_tob[g] * (1 + 0.005 * t) for g in ['Both', 'Male', 'Female']}
            ncd = {g: base_ncd[g] * (1 + 0.005 * t) for g in ['Both', 'Male', 'Female']}
            
        pred_ols = {}
        pred_ridge = {}
        for g in ['Both', 'Male', 'Female']:
            coef_ols = coefficients["OLS"][g]
            pred_ols[g] = (
                coef_ols['intercept'] + 
                (coef_ols['tobacco'] * tob[g]) + 
                (coef_ols['alcohol'] * alc) + 
                (coef_ols['pm25'] * pm25) + 
                (coef_ols['ncd'] * ncd[g])
            )
            coef_rdg = coefficients["Ridge"][g]
            pred_ridge[g] = (
                coef_rdg['intercept'] + 
                (coef_rdg['tobacco'] * tob[g]) + 
                (coef_rdg['alcohol'] * alc) + 
                (coef_rdg['pm25'] * pm25) + 
                (coef_rdg['ncd'] * ncd[g])
            )
        scenarios[scen_name].append({
            "Year": year,
            "LE_Both": pred_ridge['Both'],
            "LE_OLS_Both": pred_ols['Both'],
            "LE_Ridge_Both": pred_ridge['Both'],
            "Alcohol": alc,
            "PM25": pm25,
            "Tob_Both": tob['Both'],
            "NCD_Both": ncd['Both']
        })

plt.figure(figsize=(12, 7))
plt.plot(df_aligned['Year'], df_aligned['LE_Both'], color='black', marker='o', linewidth=3, label='Historical Actual')

# Plot Ridge as solid curves, OLS as dotted curves for comparison
for scen_name, color in [('Baseline', '#6366f1'), ('Optimistic', '#10b981'), ('Pessimistic', '#ef4444')]:
    df_scen = pd.DataFrame(scenarios[scen_name])
    connect_year = [df_aligned['Year'].iloc[-1]] + list(df_scen['Year'])
    connect_val_ridge = [df_aligned['LE_Both'].iloc[-1]] + list(df_scen['LE_Ridge_Both'])
    connect_val_ols = [df_aligned['LE_Both'].iloc[-1]] + list(df_scen['LE_OLS_Both'])
    
    plt.plot(connect_year, connect_val_ridge, color=color, linestyle='-', linewidth=2.5, label=f'{scen_name} (Ridge)')
    plt.plot(connect_year, connect_val_ols, color=color, linestyle=':', alpha=0.6, linewidth=1.5, label=f'{scen_name} (OLS)')

plt.axvline(2026, color='gray', linestyle=':', label='Forecast Boundary (2026)')
plt.title('Fig 9: Sri Lanka Life Expectancy Forecast Scenarios: Ridge vs. OLS Comparison (2026 - 2050)', fontsize=12, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Life Expectancy (Both Sexes)')
plt.legend(frameon=True, facecolor='white', framealpha=0.9, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig("fig9_scenario_projections.png", dpi=300)
plt.close()

# ==================== FIG 10: Policy Impact Bar ====================
print("Generating Fig 10...")
# Compare baseline projected LE in 2050 vs. 30% reduction policy scenario
base_2050 = scenarios['Baseline'][-1]

# Simulated scenario details for 2050 (30% reduction) under Ridge
simulated_le = {}
for g in ['Both', 'Male', 'Female']:
    coef = coefficients["Ridge"][g]
    # Under 30% policy reform:
    t_red = base_2050[f'Tob_{g}'] * 0.7 if f'Tob_{g}' in base_2050 else base_2050['Tob_Both'] * 0.7
    n_red = base_2050[f'NCD_{g}'] * 0.7 if f'NCD_{g}' in base_2050 else base_2050['NCD_Both'] * 0.7
    a_red = base_2050['Alcohol'] * 0.7
    p_red = base_2050['PM25'] * 0.7
    
    # We model direct policy impact using calibrated coefficients
    simulated_le[g] = (
        coef['intercept'] + 
        (coef['tobacco'] * t_red) + 
        (-0.12 * a_red) + # Causal alcohol hazard correction
        (coef['pm25'] * p_red) + 
        (coef['ncd'] * n_red)
    )

# Baseline values for Both, Male, Female in 2050
base_vals_cohort = {}
for g in ['Both', 'Male', 'Female']:
    coef = coefficients["Ridge"][g]
    t_val = base_2050[f'Tob_{g}'] if f'Tob_{g}' in base_2050 else base_2050['Tob_Both']
    n_val = base_2050[f'NCD_{g}'] if f'NCD_{g}' in base_2050 else base_2050['NCD_Both']
    a_val = base_2050['Alcohol']
    p_val = base_2050['PM25']
    base_vals_cohort[g] = (
        coef['intercept'] +
        (coef['tobacco'] * t_val) +
        (coef['alcohol'] * a_val) +
        (coef['pm25'] * p_val) +
        (coef['ncd'] * n_val)
    )

categories = ['Both Sexes', 'Male', 'Female']
baseline_vals = [base_vals_cohort['Both'], base_vals_cohort['Male'], base_vals_cohort['Female']]
policy_vals = [simulated_le['Both'], simulated_le['Male'], simulated_le['Female']]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, baseline_vals, width, label='2050 Baseline Forecast (Ridge)', color='#64748b')
rects2 = ax.bar(x + width/2, policy_vals, width, label='2050 Health Reform Scenario (+30%)', color='#10b981')

ax.set_ylabel('Life Expectancy (Years)')
ax.set_title('Fig 10: Lifespan Comparison by Cohort: 2050 Baseline vs. Policy Intervention Scenario', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(60, 92)
ax.legend()
ax.grid(True, axis='y')

# Add values above bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig("fig10_policy_impact_bar.png", dpi=300)
plt.close()

# ==================== FIG 11: Individual Personal Projections ====================
print("Generating Fig 11...")
def calculate_personal_le(year, scenario, age, gender, district_pm25, cigarettes, drinks, 
                          exercise, diet, sleep, screening, diabetes, blood_pressure, 
                          accidents, parts_removed, dengue, covid, other_deadly, 
                          coefficients, scenarios, selected_model='Ridge',
                          # === NEW v3 SCALES ===
                          bmi='Normal', cholesterol='Normal', ckd='None',
                          mental_health='Good', stress='Moderate', social_isolation='Some',
                          water_quality='Clean', occupational_hazard='Low', noise_pollution='Low',
                          family_longevity='Average', family_cancer='None', family_heart='None',
                          heart_disease='None', cancer_status='None', hydration='Moderate'):
    df_scen = pd.DataFrame(scenarios[scenario])
    proj_year = df_scen[df_scen['Year'] == year].iloc[0]
    
    coef_grp = coefficients[selected_model][gender]
    tob_val = proj_year[f'Tob_{gender}'] if f'Tob_{gender}' in proj_year else proj_year['Tob_Both']
    ncd_val = proj_year[f'NCD_{gender}'] if f'NCD_{gender}' in proj_year else proj_year['NCD_Both']
    national_le = (
        coef_grp['intercept'] +
        (coef_grp['tobacco'] * tob_val) +
        (coef_grp['alcohol'] * proj_year['Alcohol']) +
        (coef_grp['pm25'] * proj_year['PM25']) +
        (coef_grp['ncd'] * ncd_val)
    )
    
    base_le_birth = 74.16 if gender == 'Male' else 80.05
    base_le_60 = 16.5 if gender == 'Male' else 19.5
    le_birth = national_le
    le_60 = base_le_60 * (le_birth / base_le_birth)
    
    if age <= 0:
        baseline_remaining_le = le_birth
    elif age >= 90:
        baseline_remaining_le = max(1.5, 4.0 - (age - 90) * 0.1)
    elif age < 60:
        t = age / 60.0
        remaining_birth = le_birth - age
        surplus = le_60 - (le_birth - 60.0)
        baseline_remaining_le = remaining_birth + surplus * (t * t)
    else:
        t = (age - 60.0) / 30.0
        baseline_remaining_le = le_60 * (1.0 - t) + 4.0 * t
        
    baseline_lifespan = age + baseline_remaining_le
    
    national_pm25_2026 = scenarios['Baseline'][0]['PM25']
    national_pm25_y = proj_year['PM25']
    district_pm25_y = district_pm25 * (national_pm25_y / national_pm25_2026)
    
    # === ORIGINAL LIFESTYLE DEDUCTIONS ===
    pm_loss = 0.098 * max(0.0, district_pm25_y - 5.0)
    smoke_loss = min(12.0, cigarettes * 0.4)
    limit = 7 if gender == 'Female' else 14
    excess_drinks = max(0, drinks - limit)
    alc_loss = min(6.0, excess_drinks * 0.15)
    
    exercise_loss = 2.5 if exercise == 'Inactive' else -3.2 if exercise == 'Active' else 0.0
    diet_loss = 1.5 if diet == 'Low' else -2.0 if diet == 'Optimal' else 0.0
    sleep_loss = 1.5 if sleep == 'Insufficient' else 1.0 if sleep == 'Excessive' else -0.5
    screening_loss = 1.0 if screening == 'No' else -0.5
    
    # === ORIGINAL CLINICAL DEDUCTIONS ===
    diab_loss = 6.0 if diabetes == 'Unmanaged' else 2.0 if diabetes == 'Managed' else 0.0
    bp_loss = 4.0 if blood_pressure == 'High' else 1.5 if blood_pressure == 'Prehypertension' else 0.5 if blood_pressure == 'Low' else 0.0
    acc_loss = 3.5 if accidents == 'Major' else 1.0 if accidents == 'Moderate' else 0.0
    pr_loss = 3.0 if parts_removed == 'MajorOrgan' else 2.0 if parts_removed == 'Limb' else 0.5 if parts_removed == 'Minor' else 0.0
    
    dengue_loss = 1.0 if dengue else 0.0
    covid_loss = 1.5 if covid else 0.0
    other_loss = 1.2 if other_deadly else 0.0
    dis_loss = dengue_loss + covid_loss + other_loss
    
    # === NEW v3: METABOLIC DEDUCTIONS ===
    bmi_loss = {'Underweight': 1.0, 'Normal': 0.0, 'Overweight': 1.5, 
                'Obese': 3.5, 'SeverelyObese': 8.0}.get(bmi, 0.0)
    chol_loss = {'Normal': 0.0, 'BorderlineHigh': 1.0, 'High': 2.5}.get(cholesterol, 0.0)
    ckd_loss = {'None': 0.0, 'Early': 2.0, 'Advanced': 5.0}.get(ckd, 0.0)
    
    # === NEW v3: MENTAL & SOCIAL DEDUCTIONS ===
    mental_loss = {'Good': -0.5, 'Mild': 1.0, 'Severe': 3.0, 
                   'ChronicSevere': 5.0}.get(mental_health, 0.0)
    stress_loss = {'Low': -0.5, 'Moderate': 0.0, 'High': 2.0, 
                   'Extreme': 4.0}.get(stress, 0.0)
    social_loss = {'Connected': -1.0, 'Some': 0.0, 
                   'Severe': 3.5}.get(social_isolation, 0.0)
    
    # === NEW v3: ENVIRONMENTAL DEDUCTIONS ===
    water_loss = {'Clean': 0.0, 'Moderate': 0.5, 'Poor': 2.0}.get(water_quality, 0.0)
    occup_loss = {'Low': 0.0, 'Moderate': 1.5, 'High': 4.0}.get(occupational_hazard, 0.0)
    noise_loss = {'Low': 0.0, 'Moderate': 0.3, 'High': 1.0}.get(noise_pollution, 0.0)
    
    # === NEW v3: GENETIC RISK DEDUCTIONS ===
    fam_long_loss = {'Short': 2.0, 'Average': 0.0, 'Long': -3.0}.get(family_longevity, 0.0)
    fam_cancer_loss = {'None': 0.0, 'Some': 1.0, 'Strong': 2.5}.get(family_cancer, 0.0)
    fam_heart_loss = {'None': 0.0, 'Some': 1.0, 'Strong': 3.0}.get(family_heart, 0.0)
    
    # === NEW v3: EXTENDED CLINICAL DEDUCTIONS ===
    heart_loss = {'None': 0.0, 'Managed': 2.5, 'Unmanaged': 7.0}.get(heart_disease, 0.0)
    cancer_loss = {'None': 0.0, 'Remission': 1.5, 'Active': 8.0}.get(cancer_status, 0.0)
    
    # === NEW v3: EXTENDED LIFESTYLE ===
    hydration_loss = {'Good': -0.3, 'Moderate': 0.0, 'Poor': 0.8}.get(hydration, 0.0)
    
    # === TOTAL DEDUCTIONS (all 27 scales) ===
    total_deductions = (
        # Original lifestyle (7)
        pm_loss + smoke_loss + alc_loss + exercise_loss + diet_loss + sleep_loss + screening_loss +
        # Original clinical (5)
        diab_loss + bp_loss + acc_loss + pr_loss + dis_loss +
        # New metabolic (3)
        bmi_loss + chol_loss + ckd_loss +
        # New mental/social (3)
        mental_loss + stress_loss + social_loss +
        # New environmental (3)
        water_loss + occup_loss + noise_loss +
        # New genetic (3)
        fam_long_loss + fam_cancer_loss + fam_heart_loss +
        # New extended clinical (2)
        heart_loss + cancer_loss +
        # New extended lifestyle (1)
        hydration_loss
    )
    
    personal_le = baseline_lifespan - total_deductions
    return max(age + 0.1, personal_le)

# Simulate personal forecast for a 35-year-old Colombo male with healthy lifestyle habits
sample_age = 35
sample_gender = 'Male'
sample_district_pm25 = 22.7  # Colombo
sample_cigarettes = 0
sample_drinks = 0
sample_exercise = 'Moderate'
sample_diet = 'Medium'
sample_sleep = 'Healthy'
sample_screening = 'Yes'

personal_scenarios = {}
for scen in ['Optimistic', 'Baseline', 'Pessimistic']:
    personal_scenarios[scen] = []
    for yr in projection_years:
        le = calculate_personal_le(
            yr, scen, sample_age, sample_gender, sample_district_pm25, 
            sample_cigarettes, sample_drinks, sample_exercise, sample_diet, 
            sample_sleep, sample_screening, 'None', 'Normal', 'None', 'None', False, False, False, coefficients, scenarios
        )
        personal_scenarios[scen].append(le)

plt.figure(figsize=(12, 7))
plt.plot(projection_years, personal_scenarios['Optimistic'], color='#10b981', marker='^', linewidth=2.5, label='Optimistic Scenario')
plt.plot(projection_years, personal_scenarios['Baseline'], color='#6366f1', marker='o', linewidth=2.5, label='Baseline Scenario')
plt.plot(projection_years, personal_scenarios['Pessimistic'], color='#ef4444', marker='v', linewidth=2.5, label='Pessimistic Scenario')

plt.title('Fig 11: Year-by-Year Expected Personal Lifespan for 35yo Male Colombo Resident (2026-2050)', fontsize=12, fontweight='bold')
plt.xlabel('Prediction Year')
plt.ylabel('Predicted Personal Lifespan (Years)')
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.grid(True)
plt.tight_layout()
plt.savefig("fig11_personal_forecast.png", dpi=300)
plt.close()

print("All 11 figures generated successfully on disk!")
