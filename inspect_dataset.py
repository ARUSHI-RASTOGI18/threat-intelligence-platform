"""
Pre-flight Dataset Inspector
Location: ./inspect_dataset.py
Purpose: Inspects any input CSV, dynamically checks schema compatibility,
         computes missing rates, and confirms temporal boundaries.
"""

import os
import sys
import pandas as pd
import numpy as np

def inspect(file_path: str):
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found at path: {file_path}")
        print("Please place your CSV dataset in 'data/raw/' and update the path.")
        sys.exit(1)

    print(f"\n==================================================")
    print(f"       GTI-ARP DATASET PRE-FLIGHT INSPECTION      ")
    print(f"==================================================")
    print(f"Target File: {file_path}\n")

    # Read minimal rows first to inspect columns quickly
    try:
        df_sample = pd.read_csv(file_path, nrows=100, low_memory=False, encoding="latin1")
    except UnicodeDecodeError:
        df_sample = pd.read_csv(file_path, nrows=100, low_memory=False, encoding="utf-8")

    total_cols = len(df_sample.columns)
    print(f"Total Columns Detected: {total_cols}")

    # Full read with selected columns or sample for deep inspection
    print("Reading dataset summary statistics...")
    try:
        df = pd.read_csv(file_path, low_memory=False, encoding="latin1")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, low_memory=False, encoding="utf-8")

    rows, cols = df.shape
    print(f"Dataset Dimensions: {rows:,} rows | {cols} columns")

    # Schema mapping candidate check
    canonical_candidates = {
        "year": ["iyear", "year", "incident_year", "Year"],
        "month": ["imonth", "month", "Month"],
        "day": ["iday", "day", "Day"],
        "country": ["country_txt", "country", "Country", "CountryName"],
        "region": ["region_txt", "region", "Region"],
        "latitude": ["latitude", "lat", "Latitude"],
        "longitude": ["longitude", "lon", "lng", "Longitude"],
        "attack_type": ["attacktype1_txt", "attacktype", "attack_type", "AttackType"],
        "target_type": ["targtype1_txt", "targettype", "target_type", "TargetType"],
        "weapon_type": ["weaptype1_txt", "weapontype", "weapon_type", "WeaponType"],
        "group_name": ["gname", "group", "terrorist_group", "Group"],
        "fatalities": ["nkill", "fatalities", "killed", "Deaths"],
        "injured": ["nwound", "injuries", "wounded", "Injured"],
        "success": ["success", "is_success", "Success"],
        "suicide": ["suicide", "is_suicide", "Suicide"]
    }

    print("\n--- SCHEMA MATCHING STATUS ---")
    matched_mapping = {}
    for standard_name, possible_names in canonical_candidates.items():
        found = [col for col in possible_names if col in df.columns]
        if found:
            matched_mapping[standard_name] = found[0]
            print(f"  [OK] Standard Field: '{standard_name:<15}' -> Matched: '{found[0]}'")
        else:
            print(f"  [--] Standard Field: '{standard_name:<15}' -> NOT FOUND (Will default/impute)")

    # Temporal boundaries detection
    year_col = matched_mapping.get("year")
    if year_col:
        valid_years = pd.to_numeric(df[year_col], errors="coerce").dropna()
        min_yr = int(valid_years.min())
        max_yr = int(valid_years.max())
        print(f"\nDetected Temporal Coverage: {min_yr} to {max_yr}")
    else:
        print("\n[WARNING] No explicit year column detected.")

    print("\n==================================================")
    print("Inspection complete. Proceed to run 'python train_pipeline.py'")
    print("==================================================\n")

if __name__ == "__main__":
    target_csv = os.path.join("data", "raw", "dataset.csv")
    if len(sys.argv) > 1:
        target_csv = sys.argv[1]
    inspect(target_csv)