"""
AyuSanka NoSQL Database Manager (TinyDB)
=========================================
Stores all model data, predictions, and scale definitions as JSON documents.
Database file: ayusanka_db.json
"""

import os
import json
from datetime import datetime
from tinydb import TinyDB, Query

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ayusanka_db.json")


def init_db(path=None):
    """Initialize and return TinyDB instance."""
    db = TinyDB(path or DB_PATH, indent=4, sort_keys=False)
    return db


def store_metadata(db, title, description, source, version="3.0"):
    """Store or update model metadata."""
    table = db.table("metadata")
    table.truncate()
    table.insert({
        "title": title,
        "description": description,
        "source": source,
        "version": version,
        "lastUpdated": datetime.now().isoformat(),
        "totalScales": 27
    })
    print(f"  [DB] Metadata stored (v{version}).")


def store_coefficients(db, coefficients):
    """Store OLS and Ridge regression coefficients."""
    table = db.table("coefficients")
    table.truncate()
    for model_type, groups in coefficients.items():
        for group, coefs in groups.items():
            table.insert({
                "model": model_type,
                "cohort": group,
                **{k: float(v) for k, v in coefs.items()}
            })
    print(f"  [DB] Coefficients stored ({len(table.all())} records).")


def store_historical(db, historical_records):
    """Store historical year-by-year health indicator records."""
    table = db.table("historical")
    table.truncate()
    for record in historical_records:
        table.insert(record)
    print(f"  [DB] Historical data stored ({len(table.all())} years).")


def store_projections(db, scenarios):
    """Store projection scenarios (Baseline, Optimistic, Pessimistic)."""
    for scenario_name, records in scenarios.items():
        table_name = f"projections_{scenario_name.lower()}"
        table = db.table(table_name)
        table.truncate()
        for record in records:
            table.insert(record)
        print(f"  [DB] Projections '{scenario_name}' stored ({len(table.all())} years).")


def store_districts(db, districts):
    """Store district PM2.5 mapping data."""
    table = db.table("districts")
    table.truncate()
    for district in districts:
        table.insert(district)
    print(f"  [DB] Districts stored ({len(table.all())} entries).")


def store_scale_definitions(db):
    """Store all 27 scale definitions with metadata and impact ranges."""
    table = db.table("scale_definitions")
    table.truncate()

    scales = [
        # === Original Macro Model Scales (4) ===
        {"id": "tobacco", "category": "Macro Model", "name": "Tobacco Prevalence",
         "type": "continuous", "unit": "%", "source": "WHO GATS",
         "description": "National tobacco prevalence used in macro regression model"},
        {"id": "alcohol", "category": "Macro Model", "name": "Alcohol Consumption",
         "type": "continuous", "unit": "L/capita", "source": "WHO GHO",
         "description": "Recorded alcohol per capita used in macro regression model"},
        {"id": "pm25", "category": "Macro Model", "name": "PM2.5 Air Pollution",
         "type": "continuous", "unit": "µg/m³", "source": "WHO Ambient Air",
         "description": "National PM2.5 concentration used in macro regression model"},
        {"id": "ncd", "category": "Macro Model", "name": "NCD Mortality Probability",
         "type": "continuous", "unit": "%", "source": "WHO NCD",
         "description": "Probability of dying from NCD between 30-70 years"},

        # === Original Personal Deduction Scales (8) ===
        {"id": "smoking_personal", "category": "Lifestyle", "name": "Personal Smoking",
         "type": "slider", "range": "0-40 cigarettes/day", "source": "CDC Epidemiology",
         "impact": "-0.4 years per cigarette/day (max -12.0)", "max_impact": -12.0},
        {"id": "alcohol_personal", "category": "Lifestyle", "name": "Personal Alcohol Intake",
         "type": "slider", "range": "0-30 drinks/week", "source": "Lancet Alcohol Study",
         "impact": "-0.15 years per excess drink/week (max -6.0)", "max_impact": -6.0},
        {"id": "exercise", "category": "Lifestyle", "name": "Physical Activity Level",
         "type": "categorical", "values": ["Inactive", "Moderate", "Active"],
         "source": "WHO Guidelines", "impact": "-2.5 to +3.2 years"},
        {"id": "diet", "category": "Lifestyle", "name": "Diet Quality (Fruit & Veg)",
         "type": "categorical", "values": ["Low", "Medium", "Optimal"],
         "source": "GBD Studies", "impact": "-1.5 to +2.0 years"},
        {"id": "sleep", "category": "Lifestyle", "name": "Sleep Duration",
         "type": "categorical", "values": ["Insufficient", "Healthy", "Excessive"],
         "source": "Sleep Foundation", "impact": "-1.5 to +0.5 years"},
        {"id": "screening", "category": "Lifestyle", "name": "Annual Medical Checkup",
         "type": "boolean", "source": "MOH Sri Lanka",
         "impact": "-1.0 (no) to +0.5 (yes) years"},
        {"id": "pm25_district", "category": "Environmental", "name": "District Air Quality",
         "type": "derived", "source": "AQLI Chicago",
         "impact": "-0.098 years per µg/m³ above WHO threshold"},
        {"id": "age_gender", "category": "Demographic", "name": "Age & Gender Baseline",
         "type": "derived", "source": "WHO Life Tables Sri Lanka",
         "description": "Baseline remaining LE derived from national actuarial tables"},

        # === Original Clinical Scales (5) ===
        {"id": "diabetes", "category": "Clinical", "name": "Diabetes History",
         "type": "categorical", "values": ["None", "Managed", "Unmanaged"],
         "source": "IDF Diabetes Atlas", "impact": "0 to -6.0 years"},
        {"id": "blood_pressure", "category": "Clinical", "name": "Blood Pressure Status",
         "type": "categorical", "values": ["Normal", "Low", "Prehypertension", "High"],
         "source": "AHA Guidelines", "impact": "-0.5 to -4.0 years"},
        {"id": "accidents", "category": "Clinical", "name": "Previous Accidents",
         "type": "categorical", "values": ["None", "Moderate", "Major"],
         "source": "WHO Injury Reports", "impact": "0 to -3.5 years"},
        {"id": "parts_removed", "category": "Clinical", "name": "Body Parts Removed",
         "type": "categorical", "values": ["None", "Minor", "Limb", "MajorOrgan"],
         "source": "Surgical Outcomes Literature", "impact": "0 to -3.0 years"},
        {"id": "diseases", "category": "Clinical", "name": "Severe Past Infections",
         "type": "multi-check", "values": ["Dengue", "COVID-19", "Other"],
         "source": "WHO Infection Reports", "impact": "-1.0 to -3.7 years combined"},

        # === NEW: Metabolic Scales (3) ===
        {"id": "bmi", "category": "Metabolic", "name": "Body Mass Index (BMI)",
         "type": "categorical",
         "values": ["Underweight (<18.5)", "Normal (18.5-24.9)", "Overweight (25-29.9)",
                    "Obese (30-34.9)", "Severely Obese (35+)"],
         "source": "WHO GBD, Lancet BMI Meta-analysis",
         "impact": {"Underweight": -1.0, "Normal": 0, "Overweight": -1.5,
                    "Obese": -3.5, "SeverelyObese": -8.0}},
        {"id": "cholesterol", "category": "Metabolic", "name": "Cholesterol Level",
         "type": "categorical", "values": ["Normal", "Borderline High", "High"],
         "source": "AHA, Framingham Heart Study",
         "impact": {"Normal": 0, "BorderlineHigh": -1.0, "High": -2.5}},
        {"id": "ckd", "category": "Metabolic", "name": "Chronic Kidney Disease",
         "type": "categorical", "values": ["None", "Early Stage (1-3)", "Advanced (4-5)"],
         "source": "KDIGO Guidelines",
         "impact": {"None": 0, "Early": -2.0, "Advanced": -5.0}},

        # === NEW: Mental & Social Scales (3) ===
        {"id": "mental_health", "category": "Mental", "name": "Mental Health Status",
         "type": "categorical",
         "values": ["Good", "Mild Anxiety/Depression", "Severe", "Chronic Severe"],
         "source": "WHO Mental Health Gap, Lancet Psychiatry",
         "impact": {"Good": 0.5, "Mild": -1.0, "Severe": -3.0, "ChronicSevere": -5.0}},
        {"id": "stress", "category": "Mental", "name": "Chronic Stress Level",
         "type": "categorical", "values": ["Low", "Moderate", "High", "Extreme"],
         "source": "APA Stress Report, Lancet",
         "impact": {"Low": 0.5, "Moderate": 0, "High": -2.0, "Extreme": -4.0}},
        {"id": "social_isolation", "category": "Mental", "name": "Social Connection",
         "type": "categorical", "values": ["Well Connected", "Some Isolation", "Severe Isolation"],
         "source": "Holt-Lunstad Meta-analysis (2015)",
         "impact": {"Connected": 1.0, "Some": 0, "Severe": -3.5}},

        # === NEW: Environmental Scales (3) ===
        {"id": "water_quality", "category": "Environmental", "name": "Drinking Water Quality",
         "type": "categorical", "values": ["Clean (Pipe/Filter)", "Moderate (Well)", "Poor (Contaminated)"],
         "source": "WHO Water Safety Plans",
         "impact": {"Clean": 0, "Moderate": -0.5, "Poor": -2.0}},
        {"id": "occupational_hazard", "category": "Environmental", "name": "Occupational Hazard Exposure",
         "type": "categorical",
         "values": ["None/Low (Office/Remote)", "Moderate (Dust/Chemical)", "High (Mining/Asbestos/Heavy)"],
         "source": "ILO, NIOSH Occupational Studies",
         "impact": {"Low": 0, "Moderate": -1.5, "High": -4.0}},
        {"id": "noise_pollution", "category": "Environmental", "name": "Chronic Noise Exposure",
         "type": "categorical", "values": ["Low (<55 dB)", "Moderate (55-65 dB)", "High (>65 dB chronic)"],
         "source": "EEA, WHO Noise Guidelines",
         "impact": {"Low": 0, "Moderate": -0.3, "High": -1.0}},

        # === NEW: Genetic Risk Scales (3) ===
        {"id": "family_longevity", "category": "Genetic", "name": "Family Longevity History",
         "type": "categorical",
         "values": ["Short-lived (<65 avg)", "Average (65-80)", "Long-lived (80+)"],
         "source": "Twin Studies, GWAS Longevity",
         "impact": {"Short": -2.0, "Average": 0, "Long": 3.0}},
        {"id": "family_cancer", "category": "Genetic", "name": "Family Cancer History",
         "type": "categorical", "values": ["None Known", "Some (1 relative)", "Strong (2+ relatives)"],
         "source": "NCI, BRCA Studies",
         "impact": {"None": 0, "Some": -1.0, "Strong": -2.5}},
        {"id": "family_heart", "category": "Genetic", "name": "Family Heart Disease History",
         "type": "categorical", "values": ["None Known", "Some (1 relative)", "Strong (2+ relatives)"],
         "source": "Framingham Heart Study, AHA",
         "impact": {"None": 0, "Some": -1.0, "Strong": -3.0}},

        # === NEW: Extended Clinical Scales (2) ===
        {"id": "heart_disease", "category": "Clinical Extended", "name": "Heart Disease Status",
         "type": "categorical", "values": ["None", "Managed (Medication/Stent)", "Unmanaged/Severe"],
         "source": "AHA, ESC Heart Failure Guidelines",
         "impact": {"None": 0, "Managed": -2.5, "Unmanaged": -7.0}},
        {"id": "cancer_status", "category": "Clinical Extended", "name": "Personal Cancer History",
         "type": "categorical",
         "values": ["None", "Remission (>5 years)", "Active/Recent Treatment"],
         "source": "SEER, NCI Cancer Statistics",
         "impact": {"None": 0, "Remission": -1.5, "Active": -8.0}},

        # === NEW: Extended Lifestyle Scale (1) ===
        {"id": "hydration", "category": "Lifestyle Extended", "name": "Daily Hydration Level",
         "type": "categorical",
         "values": ["Good (8+ glasses/day)", "Moderate (4-7 glasses)", "Poor (<4 glasses)"],
         "source": "NHANES, Lancet Hydration Study",
         "impact": {"Good": 0.3, "Moderate": 0, "Poor": -0.8}},
    ]

    for scale in scales:
        table.insert(scale)

    print(f"  [DB] Scale definitions stored ({len(scales)} scales across {len(set(s['category'] for s in scales))} categories).")
    return scales


def store_prediction(db, inputs, result, scenario="Baseline"):
    """Log an individual prediction to the database."""
    table = db.table("predictions")
    record = {
        "timestamp": datetime.now().isoformat(),
        "scenario": scenario,
        "inputs": inputs,
        "result": result
    }
    table.insert(record)
    return record


def store_model_data(db, coefficients, historical_records, scenarios, districts_list,
                     title="AyuSanka Model Database v3",
                     description="Sri Lankan Life Expectancy: 27-scale super-intelligent predictor with macro regression + personal deduction engine",
                     source="WHO GHO, MOH Sri Lanka, CDC, Lancet, IDF, AHA, KDIGO, ILO, NCI, GWAS"):
    """One-shot function to store all model data into TinyDB."""
    print("\n[DB] Storing all model data to TinyDB...")
    store_metadata(db, title, description, source)
    store_coefficients(db, coefficients)
    store_historical(db, historical_records)
    store_projections(db, scenarios)
    store_districts(db, districts_list)
    store_scale_definitions(db)
    print(f"[DB] All data stored successfully to: {DB_PATH}\n")


def get_historical(db):
    """Retrieve all historical records."""
    return db.table("historical").all()


def get_projections(db, scenario="Baseline"):
    """Retrieve projection records for a given scenario."""
    table_name = f"projections_{scenario.lower()}"
    return db.table(table_name).all()


def get_coefficients(db, model_type=None, cohort=None):
    """Retrieve coefficients, optionally filtered by model type and/or cohort."""
    records = db.table("coefficients").all()
    if model_type:
        records = [r for r in records if r.get("model") == model_type]
    if cohort:
        records = [r for r in records if r.get("cohort") == cohort]
    return records


def get_districts(db):
    """Retrieve all district data."""
    return db.table("districts").all()


def get_scale_definitions(db):
    """Retrieve all scale definitions."""
    return db.table("scale_definitions").all()


def get_all_predictions(db):
    """Retrieve all logged user predictions."""
    return db.table("predictions").all()


def export_full_db_json(db, filepath=None):
    """Export entire database as a single JSON file."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(DB_PATH), "ayusanka_full_export.json")

    export = {
        "metadata": db.table("metadata").all(),
        "coefficients": db.table("coefficients").all(),
        "historical": db.table("historical").all(),
        "projections": {
            "Baseline": db.table("projections_baseline").all(),
            "Optimistic": db.table("projections_optimistic").all(),
            "Pessimistic": db.table("projections_pessimistic").all()
        },
        "districts": db.table("districts").all(),
        "scale_definitions": db.table("scale_definitions").all(),
        "predictions": db.table("predictions").all()
    }

    with open(filepath, "w") as f:
        json.dump(export, f, indent=2)

    print(f"[DB] Full database exported to: {filepath}")
    return filepath


if __name__ == "__main__":
    print("AyuSanka DB Manager — Standalone Test")
    db = init_db()
    store_scale_definitions(db)
    scales = get_scale_definitions(db)
    print(f"Total scales in DB: {len(scales)}")
    for s in scales:
        print(f"  [{s['category']}] {s['name']}")
    db.close()
