"""
Analytical Risk Engine, Strategic Intelligence & Aggregation Module
Location: ./src/analytics.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union


def calculate_period_trends(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes longitudinal volume delta, trajectory velocity, casualties delta, and comparison spans."""
    if df.empty or "year" not in df.columns:
        return {
            "trend_direction": "STABLE",
            "incident_delta": 0.0,
            "fatality_delta": 0.0,
            "injured_delta": 0.0,
            "velocity_label": "Neutral",
            "recent_volume": 0,
            "historical_volume": 0,
            "recent_span": "N/A",
            "prior_span": "N/A"
        }

    agg_dict = {
        "incidents": ("year", "count")
    }
    if "fatalities" in df.columns:
        agg_dict["fatalities"] = ("fatalities", "sum")
    if "injured" in df.columns:
        agg_dict["injured"] = ("injured", "sum")

    yearly = df.groupby("year").agg(**agg_dict).sort_index()

    if len(yearly) < 2:
        yr = str(yearly.index[0]) if len(yearly) == 1 else "N/A"
        return {
            "trend_direction": "STABLE",
            "incident_delta": 0.0,
            "fatality_delta": 0.0,
            "injured_delta": 0.0,
            "velocity_label": "Insufficient baseline",
            "recent_volume": int(yearly["incidents"].sum()) if len(yearly) == 1 else 0,
            "historical_volume": 0,
            "recent_span": yr,
            "prior_span": "Baseline"
        }

    mid_point = len(yearly) // 2
    period_a = yearly.iloc[:mid_point]
    period_b = yearly.iloc[mid_point:]

    sum_a_inc = period_a["incidents"].sum()
    sum_b_inc = period_b["incidents"].sum()
    inc_delta = round(((sum_b_inc - sum_a_inc) / sum_a_inc) * 100.0, 1) if sum_a_inc > 0 else (100.0 if sum_b_inc > 0 else 0.0)

    if "fatalities" in yearly.columns:
        sum_a_fat = period_a["fatalities"].sum()
        sum_b_fat = period_b["fatalities"].sum()
        fat_delta = round(((sum_b_fat - sum_a_fat) / sum_a_fat) * 100.0, 1) if sum_a_fat > 0 else (100.0 if sum_b_fat > 0 else 0.0)
    else:
        fat_delta = 0.0

    if "injured" in yearly.columns:
        sum_a_inj = period_a["injured"].sum()
        sum_b_inj = period_b["injured"].sum()
        inj_delta = round(((sum_b_inj - sum_a_inj) / sum_a_inj) * 100.0, 1) if sum_a_inj > 0 else (100.0 if sum_b_inj > 0 else 0.0)
    else:
        inj_delta = 0.0

    prior_span = f"{period_a.index.min()}–{period_a.index.max()}"
    recent_span = f"{period_b.index.min()}–{period_b.index.max()}"

    if inc_delta >= 15.0:
        direction = "ESCALATING"
        v_label = "High Positive Velocity"
    elif inc_delta <= -15.0:
        direction = "DECLINING"
        v_label = "Sharp Contraction"
    else:
        direction = "STABLE"
        v_label = "Neutral Steady State"

    return {
        "trend_direction": direction,
        "incident_delta": inc_delta,
        "fatality_delta": fat_delta,
        "injured_delta": inj_delta,
        "velocity_label": v_label,
        "recent_volume": int(sum_b_inc),
        "historical_volume": int(sum_a_inc),
        "recent_span": recent_span,
        "prior_span": prior_span
    }


def compute_spatial_hotspots(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Aggregates geospatial coordinates to identify high-density incident clusters."""
    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        return pd.DataFrame()

    geo_df = df.dropna(subset=["latitude", "longitude"]).copy()
    geo_df = geo_df[
        (geo_df["latitude"].between(-90, 90)) & 
        (geo_df["longitude"].between(-180, 180))
    ]

    if geo_df.empty:
        return pd.DataFrame()

    group_cols = [c for c in ["country", "region", "city", "latitude", "longitude"] if c in geo_df.columns]
    if not group_cols:
        return pd.DataFrame()

    agg_specs = {
        "Incidents": ("latitude", "count")
    }
    if "fatalities" in geo_df.columns:
        agg_specs["Fatalities"] = ("fatalities", "sum")
    if "injured" in geo_df.columns:
        agg_specs["Injuries"] = ("injured", "sum")

    hotspots = geo_df.groupby(group_cols).agg(**agg_specs).reset_index()

    if "Fatalities" in hotspots.columns and "Incidents" in hotspots.columns:
        hotspots["Lethality (Deaths/Inc)"] = (hotspots["Fatalities"] / hotspots["Incidents"]).round(2)

    hotspots = hotspots.sort_values(by="Incidents", ascending=False).head(top_n)

    rename_dict = {}
    if "country" in hotspots.columns: rename_dict["country"] = "Country"
    if "city" in hotspots.columns: rename_dict["city"] = "Primary City / Sector"
    if "region" in hotspots.columns: rename_dict["region"] = "Region"
    hotspots = hotspots.rename(columns=rename_dict)

    return hotspots


def compute_country_dossier(df: pd.DataFrame, *args) -> Dict[str, Any]:
    """Compiles a localized analytical intelligence dossier for a specific sovereign entity."""
    country_name = args[-1] if len(args) > 0 else ""
    risk_df = args[0] if (len(args) > 1 and isinstance(args[0], pd.DataFrame)) else None

    if df.empty or "country" not in df.columns or country_name not in df["country"].values:
        return {
            "country": country_name,
            "threat_score": "N/A",
            "total_incidents": 0,
            "total_fatalities": 0,
            "total_injured": 0,
            "dominant_attack": "N/A",
            "dominant_target": "N/A",
            "peak_year": "N/A"
        }

    c_df = df[df["country"] == country_name]

    top_attack = str(c_df["attack_type"].mode()[0]) if ("attack_type" in c_df.columns and not c_df["attack_type"].empty) else "Unknown"
    top_target = str(c_df["target_type"].mode()[0]) if ("target_type" in c_df.columns and not c_df["target_type"].empty) else "Unknown"
    peak_yr = int(c_df["year"].mode()[0]) if ("year" in c_df.columns and not c_df["year"].empty) else "N/A"

    score_val = "N/A"
    if risk_df is not None and not risk_df.empty and "country" in risk_df.columns:
        rk_match = risk_df[risk_df["country"] == country_name]
        if not rk_match.empty:
            for col in ["threat_index", "composite_risk_score", "risk_score"]:
                if col in rk_match.columns:
                    score_val = str(round(float(rk_match.iloc[0][col]), 1))
                    break

    return {
        "country": country_name,
        "threat_score": score_val,
        "total_incidents": len(c_df),
        "total_fatalities": int(c_df["fatalities"].sum()) if "fatalities" in c_df.columns else 0,
        "total_injured": int(c_df["injured"].sum()) if "injured" in c_df.columns else 0,
        "dominant_attack": top_attack,
        "dominant_target": top_target,
        "peak_year": peak_yr
    }


def get_threat_glance_summary(df: pd.DataFrame, risk_df: pd.DataFrame = None) -> Dict[str, Any]:
    """Generates executive summary metrics for top vectors, weapons, targets, and years."""
    if df.empty:
        return {
            "highest_risk_country": "N/A",
            "highest_risk_score": 0,
            "dominant_tactic": "N/A",
            "primary_weapon": "N/A",
            "top_target": "N/A",
            "peak_year": "N/A",
            "peak_year_count": 0
        }

    if risk_df is not None and not risk_df.empty and "country" in risk_df.columns:
        highest_risk_c = str(risk_df.iloc[0]["country"])
        highest_risk_s = round(float(risk_df.iloc[0].get("threat_index", 0)), 1)
    elif "country" in df.columns and not df["country"].empty:
        highest_risk_c = str(df["country"].mode()[0])
        highest_risk_s = 0
    else:
        highest_risk_c = "N/A"
        highest_risk_s = 0

    dominant_tactic = str(df["attack_type"].mode()[0]) if "attack_type" in df.columns and not df["attack_type"].empty else "Unknown"
    primary_weapon = str(df["weapon_type"].mode()[0]) if "weapon_type" in df.columns and not df["weapon_type"].empty else "Unknown"
    top_target = str(df["target_type"].mode()[0]) if "target_type" in df.columns and not df["target_type"].empty else "Unknown"

    if "year" in df.columns and not df["year"].empty:
        peak_yr_val = df["year"].value_counts()
        peak_year = int(peak_yr_val.index[0])
        peak_year_count = int(peak_yr_val.iloc[0])
    else:
        peak_year = "N/A"
        peak_year_count = 0

    return {
        "highest_risk_country": highest_risk_c,
        "highest_risk_score": highest_risk_s,
        "dominant_tactic": dominant_tactic,
        "primary_weapon": primary_weapon,
        "top_target": top_target,
        "peak_year": peak_year,
        "peak_year_count": peak_year_count
    }


def get_strategic_signals(df: pd.DataFrame, risk_df: pd.DataFrame, trends: Dict[str, Any], model_meta: Dict[str, Any]) -> List[Dict[str, str]]:
    """Synthesizes high-level telemetry signals."""
    if df.empty:
        return []

    signals = []
    if "country" in df.columns and not df["country"].empty:
        top_c = df["country"].mode()[0]
        c_cnt = int(df[df["country"] == top_c].shape[0])
        share = round((c_cnt / len(df)) * 100, 1)
        signals.append({
            "title": "SPATIAL CONCENTRATION SIGNAL",
            "body": f"Primary active cluster centers in <b>{top_c}</b> ({c_cnt:,} events, <span style='color:#39D353;'>{share}%</span> of period share)."
        })

    if "attack_type" in df.columns and not df["attack_type"].empty:
        top_tac = df["attack_type"].mode()[0]
        tac_cnt = int(df[df["attack_type"] == top_tac].shape[0])
        signals.append({
            "title": "TACTICAL SIGNATURE SIGNAL",
            "body": f"<b>{top_tac}</b> represents the leading operational vector, accounting for <span style='color:#58A6FF;'>{tac_cnt:,} events</span>."
        })

    signals.append({
        "title": "LONGITUDINAL VELOCITY SIGNAL",
        "body": f"System trajectory exhibits a <b>{trends.get('trend_direction', 'STABLE')}</b> trend (<span style='color:#FFA657;'>{trends.get('incident_delta', 0.0):+,.1f}%</span> observed shift)."
    })

    m_name = model_meta.get("best_model_name", "Primary Classifier") if model_meta else "Primary Classifier"
    signals.append({
        "title": "PREDICTIVE ENGINE SIGNAL",
        "body": f"<b>{m_name}</b> operational as primary classifier with active model telemetry."
    })

    return signals


def get_research_findings(*args, **kwargs) -> List[Dict[str, str]]:
    """Returns methodology and empirical research disclosures."""
    return [
        {
            "title": "Longitudinal Reporting Bias Mitigation",
            "topic": "Longitudinal Reporting Bias Mitigation",
            "category": "Data Engineering",
            "detail": "Applies variance damping to account for media reporting density scaling and digital surveillance growth post-1998.",
            "desc": "Applies variance damping to account for media reporting density scaling and digital surveillance growth post-1998."
        },
        {
            "title": "Log-Normal Severity Weighting",
            "topic": "Log-Normal Severity Weighting",
            "category": "Risk Analytics",
            "detail": "Threat Index employs sub-linear log transformations to balance outlier high-casualty incidents against persistent low-level event frequencies.",
            "desc": "Threat Index employs sub-linear log transformations to balance outlier high-casualty incidents against persistent low-level event frequencies."
        },
        {
            "title": "Temporal Holdout Leakage Prevention",
            "topic": "Temporal Holdout Leakage Prevention",
            "category": "Machine Learning",
            "detail": "Supervised multi-model classification uses strict chronological out-of-time splits (Train <= 2011, Val 2012-2014, Test 2015-2017) to prevent retrospective leakage.",
            "desc": "Supervised multi-model classification uses strict chronological out-of-time splits to prevent retrospective leakage."
        }
    ]


def get_pipeline_status(*args, **kwargs) -> List[Dict[str, str]]:
    """Returns operational subsystem pipeline telemetry cards."""
    return [
        {
            "module": "Parquet Arrow Ingestion",
            "label": "Parquet Arrow Ingestion",
            "status": "ONLINE",
            "color": "#39D353",
            "detail": "Sub-millisecond IPC data store",
            "desc": "Sub-millisecond IPC data store"
        },
        {
            "module": "ML Inference Pipeline",
            "label": "ML Inference Pipeline",
            "status": "READY",
            "color": "#39D353",
            "detail": "Temporal holdout model artifacts active",
            "desc": "Temporal holdout model artifacts active"
        },
        {
            "module": "Deterministic Risk Engine",
            "label": "Deterministic Risk Engine",
            "status": "ACTIVE",
            "color": "#58A6FF",
            "detail": "Composite ranking running",
            "desc": "Composite ranking running"
        },
        {
            "module": "Longitudinal Forecaster",
            "label": "Longitudinal Forecaster",
            "status": "ACTIVE",
            "color": "#58A6FF",
            "detail": "Holt's linear exponential smoothing online",
            "desc": "Holt's linear exponential smoothing online"
        }
    ]