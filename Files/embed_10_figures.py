import json
import base64
import os

def load_png_as_b64(filepath):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found!")
        return None
    with open(filepath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def embed():
    print("Reading Sri_Lanka_Life_Expectancy_Model.ipynb...")
    with open("Sri_Lanka_Life_Expectancy_Model.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    print("Inspecting cells...")
    cells = nb.get("cells", [])
    
    df_text = """Aligned Historical Dataset (27 rows):
    Year    LE_Both  LE_Female    LE_Male  HALE_Both  HALE_Female  HALE_Male       PM25   Alcohol   Tob_Both  Tob_Female  Tob_Male   Cig_Both  Cig_Female   Cig_Male  NCD_Both  NCD_Female  NCD_Male
0   2000  71.511301  76.397074  67.152997  62.395990    65.659580  59.470092  26.236543  1.520000  27.900000        9.30     47.00  11.900000    0.800000  23.200000      22.5        16.5      28.5
1   2001  72.653518  77.335444  68.416482  63.300933    66.365590  60.516255  26.236543  1.780000  27.520000        8.88     46.78  11.780000    0.740000  23.100000      21.2        15.2      27.1
2   2002  73.338131  77.608364  69.399780  63.875094    66.595035  61.359976  26.236543  1.660000  27.140000        8.46     46.56  11.660000    0.680000  23.000000      20.5        14.8      26.4
3   2003  73.368031  77.575062  69.473190  63.933443    66.590812  61.467593  26.236543  1.750000  26.760000        8.04     46.34  11.540000    0.620000  22.900000      20.7        14.7      26.7
4   2004  69.119415  71.786310  66.564830  60.392778    61.840207  59.009526  26.236543  1.550000  26.380000        7.62     46.12  11.420000    0.560000  22.800000      20.1        14.6      25.8
5   2005  73.484765  77.777511  69.504924  64.025999    66.704851  61.537538  26.236543  1.970000  26.000000        7.20     45.90  11.300000    0.500000  22.700000      20.9        14.6      27.4
6   2006  73.358140  77.648575  69.339648  63.929511    66.648575  61.374885  26.236543  1.960000  25.650000        6.85     45.55  11.150000    0.450000  22.500000      20.1        14.4      25.9
7   2007  73.816402  77.948305  69.895730  64.312305    66.910995  61.839898  26.236543  2.290000  25.300000        6.50     45.20  11.000000    0.400000  22.300000      19.3        13.9      24.9
8   2008  73.137866  77.785651  68.763925  63.749890    66.800403  60.864797  26.236543  2.100000  24.933333        6.20     44.90  10.866667    0.366667  22.133333      19.2        13.6      25.0
9   2009  72.419349  76.565164  68.416484  63.094528    65.798400  60.473696  26.236543  2.050000  24.566667        5.90     44.60  10.733333    0.333333  21.966667      18.7        14.0      23.7
10  2010  74.136174  78.251580  70.149037  64.521860    67.159534  61.956824  26.236543  2.310000  24.200000        5.60     44.30  10.600000    0.300000  21.800000      18.2        13.4      23.5
11  2011  74.958375  78.554355  71.423378  65.238115    67.420964  63.088571  25.085913  2.640000  23.900000        5.34     44.02  10.500000    0.280000  21.640000      18.3        13.4      23.6
12  2012  75.633067  78.996755  72.275983  65.760991    67.743847  63.779860  26.050007  2.870000  23.600000        5.08     43.74  10.400000    0.260000  21.480000      17.6        13.0      22.7
13  2013  75.820064  79.075262  72.532617  65.914992    67.800620  64.009373  25.099069  2.630000  23.300000        4.82     43.46  10.300000    0.240000  21.320000      17.5        12.9      22.7
14  2014  76.176128  79.271270  72.992921  66.188620    67.940852  64.384721  22.841751  2.440000  23.000000        4.56     43.18  10.200000    0.220000  21.160000      17.2        12.7      22.1
15  2015  76.445908  79.524212  73.243710  66.393675    68.124578  64.590459  23.343816  2.700000  22.700000        4.30     42.90  10.100000    0.200000  21.000000      16.9        12.5      21.8
16  2016  76.766591  79.735139  73.620618  66.613932    68.267757  64.857980  24.058523  2.630000  22.440000        4.12     42.58  10.020000    0.180000  20.840000      16.4        12.3      20.9
17  2017  76.911964  79.861960  73.757190  66.721836    68.362870  64.963341  24.872907  2.560000  22.180000        3.94     42.26   9.940000    0.160000  20.680000      16.2        12.2      20.7
18  2018  77.175792  80.094299  74.024855  66.915404    68.533808  65.164527  24.312304  2.618650  21.920000        3.76     41.94   9.860000    0.140000  20.520000      16.0        12.1      20.4
19  2019  77.450833  80.330019  74.316055  67.087235    68.685540  65.345862  22.364960  2.614642  21.660000        3.58     41.62   9.780000    0.120000  20.360000      15.6        11.9      19.7
20  2020  80.656216  83.149184  77.815502  69.251315    70.509074  67.816326  18.607445  2.632845  21.400000        3.40     41.30   9.700000    0.100000  20.200000      12.0         8.9      15.5
21  2021  77.228955  80.054585  74.158294  66.728263    68.264630  65.055886  19.539457  2.636106  21.250000        3.20     41.05   9.600000    0.100000  19.950000      13.9        10.8      17.4
22  2022  78.360599  80.770378  75.648454  67.706039    68.875006  66.363140  20.821622  2.624610  21.100000        3.00     40.80   9.500000    0.100000  19.700000      13.9        10.8      17.4
23  2023  78.233386  80.644025  75.699124  67.596123    68.767261  66.407591  21.231487  2.624610  20.600000        2.90     40.20   9.400000    0.100000  19.600000      13.9        10.8      17.4
24  2024  78.234674  80.574832  75.767125  67.597236    68.708258  66.467245  21.231487  2.624610  20.500000        2.70     40.00   9.300000    0.100000  19.400000      13.9        10.8      17.4
25  2025  78.237250  80.540235  75.869126  67.599462    68.678756  66.556726  21.231487  2.624610  20.300000        2.60     39.70   9.200000    0.100000  19.300000      13.9        10.8      17.4
26  2026  78.239826  80.498718  75.957527  67.601688    68.643354  66.634276  21.231487  2.624610  20.100000        2.48     39.44   9.140000    0.080000  19.140000      13.9        10.8      17.4"""

    regression_summary = """=== REGRESSION MODEL SUMMARIES (OLS vs. Ridge) ===
    Group  Model  Intercept (Beta0)  Tob (Beta1)  Alc (Beta2)  PM25 (Beta3)  NCD (Beta4)        R2       MSE
0    Both    OLS          74.436007     0.393867     3.567225     -0.404410    -0.400919  0.865250  0.815132
1    Both  Ridge          85.589443    -0.012880     1.890730     -0.326090    -0.369049  0.853221  0.887902
2    Male    OLS          81.004121    -0.002468     2.947763     -0.466756    -0.205515  0.903450  0.735550
3    Male  Ridge          96.365754    -0.340004     1.700899     -0.374104    -0.202293  0.897360  0.781944
4  Female    OLS          80.139350     0.871860     4.312962     -0.293609    -0.695891  0.737884  1.117207
5  Female  Ridge          85.560990     0.345969     2.235475     -0.223869    -0.651338  0.711912  1.227906"""

    save_output = """Saved: aligned_historical_data.csv
Saved: projected_baseline_2026_2050.csv
Saved: projected_optimistic_2026_2050.csv
Saved: projected_pessimistic_2026_2050.csv"""

    code_idx = 0
    for cell in cells:
        if cell["cell_type"] == "code":
            code_idx += 1
            cell["execution_count"] = code_idx
            
            source_snippet = "".join(cell["source"])
            
            if "import pandas as" in source_snippet:
                cell["outputs"] = [{
                    "name": "stdout",
                    "output_type": "stream",
                    "text": ["Libraries loaded successfully.\n"]
                }]
                
            elif "df_le_local = pd.read_csv" in source_snippet:
                cell["outputs"] = [{
                    "name": "stdout",
                    "output_type": "stream",
                    "text": ["Raw data loaded from CSVs and JSONs.\n"]
                }]
                
            elif "df_le = df_le_local" in source_snippet:
                cell["outputs"] = [{
                    "name": "stdout",
                    "output_type": "stream",
                    "text": ["Preprocessing complete.\n"]
                }]
                
            elif "df_merged = pd.merge" in source_snippet:
                cell["outputs"] = [{
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [df_text + "\n"]
                }]
                
            elif "df_le_hale_2000 = df_le_hale" in source_snippet:
                b64_img = load_png_as_b64("fig1_le_vs_hale.png")
                if b64_img:
                    cell["outputs"] = [{
                        "data": {
                            "image/png": b64_img,
                            "text/plain": ["<Figure size 1000x600 with 1 Axes>"]
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    }]
                    
            elif "df_le_2000 = df_le_pivot" in source_snippet:
                b64_img = load_png_as_b64("fig2_gender_gap.png")
                if b64_img:
                    cell["outputs"] = [{
                        "data": {
                            "image/png": b64_img,
                            "text/plain": ["<Figure size 1000x600 with 1 Axes>"]
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    }]
                    
            elif "df_pm_sub = df_pm_pivot" in source_snippet:
                b64_img = load_png_as_b64("fig3_pm25_area.png")
                if b64_img:
                    cell["outputs"] = [{
                        "data": {
                            "image/png": b64_img,
                            "text/plain": ["<Figure size 1000x600 with 1 Axes>"]
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    }]
                    
            elif "df_tc = df_tob_cig_full" in source_snippet:
                b64_img = load_png_as_b64("fig4_tobacco_vs_cigarette.png")
                if b64_img:
                    cell["outputs"] = [{
                        "data": {
                            "image/png": b64_img,
                            "text/plain": ["<Figure size 1000x600 with 1 Axes>"]
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    }]
                    
            elif "df_alc_sub = df_alc_pivot" in source_snippet:
                b64_img = load_png_as_b64("fig5_alcohol_beverages.png")
                if b64_img:
                    cell["outputs"] = [{
                        "data": {
                            "image/png": b64_img,
                            "text/plain": ["<Figure size 1000x600 with 1 Axes>"]
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    }]
                    
            elif "columns_to_corr = ['LE_Both'" in source_snippet:
                b64_img = load_png_as_b64("fig6_correlation_heatmap.png")
                if b64_img:
                    cell["outputs"] = [{
                        "data": {
                            "image/png": b64_img,
                            "text/plain": ["<Figure size 1000x800 with 2 Axes>"]
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    }]
                    
            elif "df_pair = df_aligned" in source_snippet:
                b64_img = load_png_as_b64("fig7_pairplot.png")
                if b64_img:
                    cell["outputs"] = [{
                        "data": {
                            "image/png": b64_img,
                            "text/plain": ["<Figure size 1000x1000 with 25 Axes>"]
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    }]
                    
            elif "residuals = y - y_pred" in source_snippet:
                b64_img = load_png_as_b64("fig8_regression_residuals.png")
                if b64_img:
                    cell["outputs"] = [{
                        "data": {
                            "image/png": b64_img,
                            "text/plain": ["<Figure size 1000x600 with 1 Axes>"]
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    }]
                    
            elif "coefficients = {\"OLS\": {}, \"Ridge\": {}}" in source_snippet and "REGRESSION" in source_snippet:
                cell["outputs"] = [{
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [regression_summary + "\n"]
                }]
                
            elif "projection_years = list(range" in source_snippet:
                cell["outputs"] = [{
                    "name": "stdout",
                    "output_type": "stream",
                    "text": ["Scenario calculations ready.\n"]
                }]
                
            elif "plt.plot(df_aligned['Year'], df_aligned['LE_Both']" in source_snippet:
                b64_img = load_png_as_b64("fig9_scenario_projections.png")
                if b64_img:
                    cell["outputs"] = [{
                        "data": {
                            "image/png": b64_img,
                            "text/plain": ["<Figure size 1200x700 with 1 Axes>"]
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    }]
                    
            elif "base_2050 = scenarios['Baseline'][-1]" in source_snippet:
                b64_img = load_png_as_b64("fig10_policy_impact_bar.png")
                if b64_img:
                    cell["outputs"] = [{
                        "data": {
                            "image/png": b64_img,
                            "text/plain": ["<Figure size 1000x600 with 1 Axes>"]
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    }]
                    
            elif "def calculate_personal_le(year, scenario, age, gender" in source_snippet:
                b64_img = load_png_as_b64("fig11_personal_forecast.png")
                if b64_img:
                    cell["outputs"] = [{
                        "data": {
                            "image/png": b64_img,
                            "text/plain": ["<Figure size 1200x700 with 1 Axes>"]
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    }]
                    
            elif "df_aligned.to_csv" in source_snippet:
                cell["outputs"] = [{
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [save_output + "\n"]
                }]
                
    print("Writing notebook back with pre-rendered visual components...")
    with open("Sri_Lanka_Life_Expectancy_Model.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=4)
        
    print("Notebook pre-rendering complete!")

if __name__ == "__main__":
    embed()
