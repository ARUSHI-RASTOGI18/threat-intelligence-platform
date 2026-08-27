"""
Data Ingestion, Dynamic Schema Adapter & Quality Audit Engine
Location: ./src/data_loader.py
Purpose: Ingests raw threat CSV data, resolves duplicates, computes transparent
         data quality scores, and caches standardized Parquet data with Streamlit caching.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st

CANONICAL_PRIORITY = {
    "year": ["year", "incident_year", "iyear"],
    "month": ["month", "imonth"],
    "day": ["day", "iday"],
    "country": ["country", "countryname", "country_txt"],
    "region": ["region", "region_txt"],
    "province_state": ["provstate", "province", "state", "province_state"],
    "city": ["city", "city_txt"],
    "latitude": ["lat", "latitude"],
    "longitude": ["lon", "lng", "longitude"],
    "attack_type": ["attacktype", "attack_type", "attacktype1_txt"],
    "target_type": ["targettype", "target_type", "targtype1_txt"],
    "weapon_type": ["weapontype", "weapon_type", "weaptype1_txt"],
    "group_name": ["group", "terrorist_group", "gname"],
    "fatalities": ["killed", "deaths", "fatalities", "nkill"],
    "injured": ["injuries", "wounded", "injured", "nwound"],
    "success": ["is_success", "success"],
    "suicide": ["is_suicide", "suicide"],
    "summary": ["summary_txt", "summary"]
}

def resolve_raw_file_path() -> str:
    raw_dir = os.path.join("data", "raw")
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)
    candidates = [f for f in os.listdir(raw_dir) if f.lower().endswith(".csv")]
    if candidates:
        return os.path.join(raw_dir, candidates[0])
    return os.path.join(raw_dir, "dataset.csv")

def deduplicate_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    clean_cols = [str(c).strip() for c in df.columns]
    df.columns = clean_cols
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    return df

def standardize_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_raw = deduplicate_raw_columns(df_raw)
    lower_col_map = {col.lower(): col for col in df_raw.columns}
    standardized_series_dict = {}

    for canonical_name, candidate_list in CANONICAL_PRIORITY.items():
        matched_source_col = None
        for cand in reversed(candidate_list):
            cand_lower = cand.lower()
            if cand_lower in lower_col_map:
                matched_source_col = lower_col_map[cand_lower]
                break

        if matched_source_col is not None:
            col_data = df_raw[matched_source_col]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            standardized_series_dict[canonical_name] = col_data.copy()

    df = pd.DataFrame(standardized_series_dict)

    defaults = {
        "year": 1970, "month": 1, "day": 1,
        "country": "Unknown Country", "region": "Unknown Region",
        "province_state": "Unknown State", "city": "Unknown City",
        "latitude": np.nan, "longitude": np.nan,
        "attack_type": "Unknown Attack", "target_type": "Unknown Target",
        "weapon_type": "Unknown Weapon", "group_name": "Unknown Group",
        "fatalities": 0.0, "injured": 0.0, "success": 1, "suicide": 0,
        "summary": "No historical summary logged."
    }

    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val

    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(1970).astype(int)
    df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["injured"] = pd.to_numeric(df["injured"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["success"] = pd.to_numeric(df["success"], errors="coerce").fillna(1).astype(int)
    df["suicide"] = pd.to_numeric(df["suicide"], errors="coerce").fillna(0).astype(int)

    categorical_cols = ["country", "region", "province_state", "city", "attack_type", "target_type", "weapon_type", "group_name"]
    for c in categorical_cols:
        series_obj = df[c]
        if isinstance(series_obj, pd.DataFrame):
            series_obj = series_obj.iloc[:, 0]
        fallback_txt = f"Unknown {c.replace('_', ' ').title()}"
        df[c] = series_obj.fillna(fallback_txt).astype(str).str.strip()

    df = df[df["year"] >= 1900].copy()
    return df

def generate_demo_dataset() -> pd.DataFrame:
    np.random.seed(42)
    n = 2500
    years = np.random.randint(1995, 2018, n)
    countries = ["Iraq", "Afghanistan", "Pakistan", "India", "Nigeria", "Colombia", "Philippines", "United Kingdom", "United States", "Somalia"]
    regions = ["Middle East & North Africa", "South Asia", "Sub-Saharan Africa", "South America", "Southeast Asia", "Western Europe", "North America"]
    attacks = ["Bombing/Explosion", "Armed Assault", "Assassination", "Hostage Taking (Kidnapping)", "Facility/Infrastructure Attack", "Unarmed Assault"]
    targets = ["Private Citizens & Property", "Military", "Police", "Government (General)", "Business", "Transportation", "Educational Institution"]
    weapons = ["Explosives", "Firearms", "Incendiary", "Melee", "Chemical"]
    groups = ["Taliban", "ISIL", "Boko Haram", "Al-Shabaab", "CPI-Maoist", "Unknown Group"]

    lat_map = {"Iraq": (33.3, 44.3), "Afghanistan": (33.9, 67.7), "Pakistan": (30.3, 69.3), "India": (20.5, 78.9), "Nigeria": (9.0, 8.6)}
    
    data = []
    for i in range(n):
        c = np.random.choice(countries)
        base_lat, base_lon = lat_map.get(c, (20.0, 30.0))
        data.append({
            "year": int(years[i]), "month": int(np.random.randint(1, 13)), "day": int(np.random.randint(1, 29)),
            "country": c, "region": np.random.choice(regions), "province_state": "State Area", "city": "Urban Zone",
            "latitude": float(base_lat + np.random.normal(0, 1.5)), "longitude": float(base_lon + np.random.normal(0, 1.5)),
            "attack_type": np.random.choice(attacks, p=[0.45, 0.25, 0.12, 0.08, 0.07, 0.03]),
            "target_type": np.random.choice(targets), "weapon_type": np.random.choice(weapons), "group_name": np.random.choice(groups),
            "fatalities": float(np.random.choice([0, 1, 2, 5, 12], p=[0.45, 0.3, 0.15, 0.07, 0.03])),
            "injured": float(np.random.choice([0, 1, 4, 10], p=[0.5, 0.25, 0.15, 0.1])),
            "success": int(np.random.choice([1, 0], p=[0.88, 0.12])), "suicide": int(np.random.choice([0, 1], p=[0.92, 0.08])),
            "summary": "Historical analytical incident record."
        })
    return pd.DataFrame(data)

@st.cache_data(show_spinner="Optimizing and caching dataset into memory...")
def load_analytical_data(force_reload: bool = False) -> pd.DataFrame:
    processed_dir = os.path.join("data", "processed")
    processed_parquet = os.path.join(processed_dir, "cleaned_dataset.parquet")
    
    if not force_reload and os.path.exists(processed_parquet):
        try:
            return pd.read_parquet(processed_parquet)
        except Exception:
            pass

    raw_path = resolve_raw_file_path()
    if not os.path.exists(raw_path):
        return generate_demo_dataset()

    try:
        df_raw = pd.read_csv(raw_path, low_memory=False, encoding="latin1")
    except UnicodeDecodeError:
        df_raw = pd.read_csv(raw_path, low_memory=False, encoding="utf-8")

    df_clean = standardize_dataframe(df_raw)
    
    os.makedirs(processed_dir, exist_ok=True)
    try:
        df_clean.to_parquet(processed_parquet, index=False)
    except Exception:
        pass

    return df_clean

@st.cache_data
def audit_dataset_quality(df: pd.DataFrame) -> dict:
    total_rows = len(df)
    if total_rows == 0:
        return {"data_quality_score": 0.0, "completeness": 0.0, "geo_coverage": 0.0}

    core_cols = ["year", "country", "region", "attack_type", "target_type", "weapon_type", "fatalities", "injured"]
    null_counts = {c: int(df[c].isnull().sum()) for c in df.columns}
    
    missing_cells = sum(null_counts[c] for c in core_cols if c in null_counts)
    total_cells = len(core_cols) * total_rows
    completeness = round(((total_cells - missing_cells) / total_cells) * 100.0, 2)
    
    geo_valid = int(df["latitude"].notnull().sum() & df["longitude"].notnull().sum())
    geo_coverage = round((geo_valid / total_rows) * 100.0, 2)
    
    raw_dups = int(df.duplicated(subset=core_cols).sum())
    dup_pct = round((raw_dups / total_rows) * 100.0, 2)

    data_quality_score = round(0.50 * completeness + 0.30 * geo_coverage + 0.20 * (100.0 - min(dup_pct, 20.0) * 5), 1)

    return {
        "total_records": total_rows,
        "total_columns": len(df.columns),
        "data_quality_score": min(100.0, max(0.0, data_quality_score)),
        "completeness_pct": completeness,
        "geocoding_coverage_pct": geo_coverage,
        "duplicate_rows_count": raw_dups,
        "duplicate_rows_pct": dup_pct,
        "null_counts": null_counts,
        "memory_mb": round(df.memory_usage().sum() / (1024 ** 2), 2)
    }

@st.cache_data
def get_dataset_metadata(df: pd.DataFrame) -> dict:
    min_yr = int(df["year"].min()) if not df.empty else 0
    max_yr = int(df["year"].max()) if not df.empty else 0
    return {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "min_year": min_yr,
        "max_year": max_yr,
        "coverage_label": f"{min_yr} – {max_yr}",
        "total_fatalities": int(df["fatalities"].sum()) if not df.empty else 0,
        "total_injured": int(df["injured"].sum()) if not df.empty else 0,
        "unique_countries": int(df["country"].nunique()) if not df.empty else 0,
        "unique_groups": int(df["group_name"].nunique()) if not df.empty else 0
    }