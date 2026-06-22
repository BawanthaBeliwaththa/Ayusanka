<div align="center">
  <img src="logo_full.png" alt="AyuSanka Logo" width="300" />
  
  # AyuSanka: Sri Lanka Life Expectancy Analytics & Individual Lifespan Predictor
  
  **An advanced, bi-lingual, predictive analytics dashboard designed to model both national life expectancy trends and individual lifespan outcomes in Sri Lanka.**

  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
  [![Frontend](https://img.shields.io/badge/Frontend-Vanilla_JS-f7df1e.svg)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
  [![Backend](https://img.shields.io/badge/Backend-Python_3.10-3776AB.svg)](https://www.python.org/)
  [![UI](https://img.shields.io/badge/UI-Glassmorphism-ff69b4.svg)](#)
</div>

---

## 📖 Overview

**AyuSanka** is a comprehensive predictive modeling application built by **Bawantha Beliwaththa**. It leverages historical data from the World Health Organization (WHO) and the Ministry of Health (MOH) Sri Lanka to forecast national longevity, while also providing a highly personalized individual lifespan predictor powered by epidemiological hazard ratios.

The application features a premium, responsive, glassmorphic User Interface with full bi-lingual support (English & Sinhala), designed to be deployed as a seamless, static client-side application.

---

## ✨ Key Features

### 1. 🧑‍⚕️ Individual Lifespan Predictor
A highly granular calculator that estimates your remaining life expectancy based on over **27+ comprehensive factors**:
- **Demographics & Geography**: Age, Gender, and District-level PM2.5 Air Quality index.
- **Substance & Lifestyle**: Tobacco smoking, alcohol consumption, diet quality, exercise routines, and sleep patterns.
- **Metabolic & Genetic**: BMI, cholesterol, kidney function, and familial longevity history.
- **Clinical History**: Past severe diseases (Dengue, COVID-19), removed organs, severe accidents, diabetes, and blood pressure.
- **Dynamic Output**: Generates an interactive habit-based longevity matrix showing exactly how many years you are gaining or losing from each habit.

### 2. 📈 National Forecasting (2000 - 2050)
Analyzes historical Sri Lankan health data to project future national life expectancy under three distinct scenarios: **Baseline, Optimistic, and Pessimistic**.
- Models use both **Ordinary Least Squares (OLS)** and **L2-Regularized Ridge Regression** to prevent overfitting and spurious economic correlations.
- Interactive charts and data tables breaking down projections for Male, Female, and Both Sexes.

### 3. 🏛️ Nationwide Policy Simulator
A powerful tool for public health researchers. Adjust sliders to simulate the impact of nationwide policy reforms (e.g., 20% drop in tobacco prevalence, 15% improvement in air quality) and instantly see the simulated impact on the national life expectancy by the year 2050.

### 4. 🚀 Zero-Backend Architecture
The core calculation engine is built entirely in Vanilla JavaScript and runs in the browser, meaning the application can be hosted on **any static web server** (like GitHub Pages or a VPS `public_html` folder) without requiring an active Python backend. 

### 5. 🌗 Premium UI/UX
- **Glassmorphism Design:** Beautiful, blurred, transparent interfaces.
- **Fully Responsive:** Collapsible sidebars and mobile-optimized charting.
- **Bi-Lingual:** Instant, state-based switching between English and Sinhala.
- **Dark/Light Mode:** Full theming support.
- **Data Export:** Generate and download your personalized forecast as a CSV directly from the browser.

---

## 🛠️ Technology Stack

- **Frontend:** HTML5, CSS3 (Vanilla, Custom Variables, Flexbox/Grid), JavaScript (ES6+).
- **Charting:** [ApexCharts](https://apexcharts.com/) for dynamic, interactive data visualization.
- **Data Compilation Pipeline (Optional Backend):** Python 3, `TinyDB` for NoSQL storage, and a custom data extraction script (`export_csv.py`) that pre-compiles historical data and model coefficients into static JSON/CSV artifacts.
- **Styling:** FontAwesome (Icons), Google Fonts (Outfit, Inter).

---

## 🚀 Getting Started / Deployment

Because AyuSanka is compiled into a static site, deployment is incredibly easy.

### Local Development
1. Clone the repository:
   ```bash
   git clone https://github.com/BawanthaBeliwaththa/AyuSanka.git
   cd AyuSanka/dist
   ```
2. Start a local HTTP server:
   ```bash
   python -m http.server 8000
   ```
3. Open `http://localhost:8000` in your web browser.

### Live Demo
🌐 **View the Live Application Here:** [https://beliwaththa.web.lk/Ayusanka/](https://beliwaththa.web.lk/Ayusanka/)

### VPS Deployment (Production)
1. Upload the entire contents of the `dist/` directory to your web server's `public_html` or `/var/www/html` folder.
2. The application will instantly go live—no backend configuration needed!

---

## 📊 Methodological Note

In the historical data, economic growth in Sri Lanka (2000-2021) coincided with both an increase in life expectancy and recorded alcohol consumption, creating a spurious positive correlation. To handle this rigorously:
1. **Ridge Regression (L2)** is used in the national model to penalize and shrink these confounded coefficients.
2. The **Individual Predictor** overrides macro-correlations and utilizes established causal Hazard Ratios (HR) from epidemiological literature (Lancet, WHO, CDC) to calculate direct survival risks.

---

## 👨‍💻 Author

**Bawantha Beliwaththa**  
- LinkedIn: [linkedin.com/in/beliwaththa/](https://www.linkedin.com/in/beliwaththa/)
- GitHub: [github.com/BawanthaBeliwaththa](https://github.com/BawanthaBeliwaththa)

---

> **Disclaimer:** AyuSanka is an academic and predictive analytics tool. The individual lifespan predictions are estimations based on population-level hazard ratios and do not constitute professional medical advice.
