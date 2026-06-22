"""
AyuSanka CSV Export Module
===========================
Exports all database collections to CSV files in the exports/ directory.
Can be run standalone or called from train_model.py.
"""

import os
import csv
import json
from datetime import datetime

try:
    from db_manager import init_db, get_historical, get_projections, get_coefficients, \
        get_districts, get_scale_definitions, get_all_predictions
except ImportError:
    pass

EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")


def ensure_export_dir():
    """Create exports directory if it doesn't exist."""
    os.makedirs(EXPORT_DIR, exist_ok=True)
    return EXPORT_DIR


def export_historical_csv(db=None, records=None):
    """Export historical health indicator data to CSV."""
    ensure_export_dir()
    if records is None and db is not None:
        records = get_historical(db)
    if not records:
        print("  [CSV] No historical records to export.")
        return

    filepath = os.path.join(EXPORT_DIR, "historical_data.csv")
    # Get all keys from first record, excluding TinyDB internal doc_id
    fieldnames = [k for k in records[0].keys() if k != "doc_id"]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k) for k in fieldnames})

    print(f"  [CSV] Historical data exported ({len(records)} rows) -> {filepath}")
    return filepath


def export_projections_csv(db=None, scenario="Baseline", records=None):
    """Export projection scenario data to CSV."""
    ensure_export_dir()
    if records is None and db is not None:
        records = get_projections(db, scenario)
    if not records:
        print(f"  [CSV] No projection records for '{scenario}'.")
        return

    filepath = os.path.join(EXPORT_DIR, f"projections_{scenario.lower()}.csv")
    fieldnames = [k for k in records[0].keys() if k != "doc_id"]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k) for k in fieldnames})

    print(f"  [CSV] Projections '{scenario}' exported ({len(records)} rows) -> {filepath}")
    return filepath


def export_coefficients_csv(db=None, coefficients_dict=None):
    """Export model coefficients to CSV."""
    ensure_export_dir()
    filepath = os.path.join(EXPORT_DIR, "model_coefficients.csv")

    if coefficients_dict is not None:
        # Direct from dict (train_model.py integration)
        rows = []
        for model_type, groups in coefficients_dict.items():
            for group, coefs in groups.items():
                row = {"model": model_type, "cohort": group}
                row.update({k: v for k, v in coefs.items()})
                rows.append(row)
    elif db is not None:
        rows = get_coefficients(db)
        rows = [{k: v for k, v in r.items() if k != "doc_id"} for r in rows]
    else:
        print("  [CSV] No coefficients data to export.")
        return

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  [CSV] Coefficients exported ({len(rows)} rows) -> {filepath}")
    return filepath


def export_districts_csv(db=None, districts_list=None):
    """Export districts PM2.5 data to CSV."""
    ensure_export_dir()
    if districts_list is None and db is not None:
        districts_list = get_districts(db)
        districts_list = [{k: v for k, v in d.items() if k != "doc_id"} for d in districts_list]
    if not districts_list:
        print("  [CSV] No districts data to export.")
        return

    filepath = os.path.join(EXPORT_DIR, "districts_pm25.csv")
    fieldnames = list(districts_list[0].keys())

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(districts_list)

    print(f"  [CSV] Districts exported ({len(districts_list)} rows) -> {filepath}")
    return filepath


def export_scale_definitions_csv(db=None, scales=None):
    """Export scale definitions to CSV."""
    ensure_export_dir()
    if scales is None and db is not None:
        scales = get_scale_definitions(db)

    if not scales:
        print("  [CSV] No scale definitions to export.")
        return

    filepath = os.path.join(EXPORT_DIR, "scale_definitions.csv")
    # Flatten complex fields for CSV
    rows = []
    for s in scales:
        row = {
            "id": s.get("id", ""),
            "category": s.get("category", ""),
            "name": s.get("name", ""),
            "type": s.get("type", ""),
            "source": s.get("source", ""),
            "description": s.get("description", ""),
            "values": json.dumps(s.get("values", [])) if isinstance(s.get("values"), list) else str(s.get("values", "")),
            "impact": json.dumps(s.get("impact", "")) if isinstance(s.get("impact"), dict) else str(s.get("impact", "")),
            "max_impact": s.get("max_impact", ""),
            "unit": s.get("unit", ""),
            "range": s.get("range", "")
        }
        rows.append(row)

    fieldnames = ["id", "category", "name", "type", "source", "description", "values", "impact", "max_impact", "unit", "range"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  [CSV] Scale definitions exported ({len(rows)} rows) -> {filepath}")
    return filepath


def export_predictions_csv(db=None, predictions=None):
    """Export user prediction logs to CSV."""
    ensure_export_dir()
    if predictions is None and db is not None:
        predictions = get_all_predictions(db)

    if not predictions:
        print("  [CSV] No prediction records to export (none logged yet).")
        return

    filepath = os.path.join(EXPORT_DIR, "user_predictions.csv")

    # Flatten nested inputs/result dicts
    rows = []
    for p in predictions:
        row = {
            "timestamp": p.get("timestamp", ""),
            "scenario": p.get("scenario", ""),
        }
        inputs = p.get("inputs", {})
        for k, v in inputs.items():
            row[f"input_{k}"] = v
        result = p.get("result", {})
        for k, v in result.items():
            row[f"result_{k}"] = v
        rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    # Collect all possible keys across rows
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  [CSV] Predictions exported ({len(rows)} rows) -> {filepath}")
    return filepath


def export_all(db=None, coefficients_dict=None, historical_records=None,
               scenarios=None, districts_list=None):
    """Export all data to CSV files."""
    print("\n[CSV] Exporting all data to CSV files...")
    ensure_export_dir()

    export_historical_csv(db, historical_records)

    for scenario in ["Baseline", "Optimistic", "Pessimistic"]:
        scen_records = scenarios.get(scenario) if scenarios else None
        export_projections_csv(db, scenario, scen_records)

    export_coefficients_csv(db, coefficients_dict)
    export_districts_csv(db, districts_list)
    export_scale_definitions_csv(db)
    export_predictions_csv(db)

    print(f"[CSV] All exports complete -> {EXPORT_DIR}/\n")
    return EXPORT_DIR


if __name__ == "__main__":
    print("AyuSanka CSV Export — Standalone")
    print(f"Export directory: {EXPORT_DIR}")

    db = init_db()
    export_all(db=db)
    db.close()
    print("Done.")
