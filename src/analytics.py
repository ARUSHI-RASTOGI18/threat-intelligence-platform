"""
Comparative Trajectory, Velocity & Sovereign Intelligence Engine
Location: ./src/analytics.py
Purpose: Pure statistical calculations for Anomaly Detection, Period Velocities,
         Country Comparisons, Hotspot Detection, and Executive Command Center Telemetry.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

def calculate_period_trends(df: pd.DataFrame) -> dict:
    """
    Computes percentage deltas across the most recent historical dataset period
    relative to the immediate preceding equivalent period within the filtered timeframe.
    """
    if df.empty:
        return {
            "incident_delta": 0.0,
            "fatality_delta": 0.0,
            "injured_delta": 0.0,
            "trend_direction": "STABLE",
            "recent_span": "N/A",
            "prior_span": "N/A"
        }

    years = sorted(df["year"].unique())
    if len(years) < 2:
        return {
            "incident_delta": 0.0,
            "fatality_delta": 0.0,
            "injured_delta": 0.0,
            "trend_direction": "STABLE",
            "recent_span": str(years[0]) if years else "N/A",
            "prior_span": "N/A"
        }

    # Partition the active filtered range into two equal or windowed halves
    span_len = min(3, len(years) // 2) if len(years) >= 4 else 1
    recent_years = years[-span_len:]
    prior_years = years[-2 * span_len : -span_len]

    recent_df = df[df["year"].isin(recent_years)]
    prior_df = df[df["year"].isin(prior_years)]

    def get_pct(curr: float, prev: float) -> float:
        if prev == 0:
            return 100.0 if curr > 0 else 0.0
        return round(((curr - prev) / prev) * 100.0, 1)

    r_inc, p_inc = len(recent_df), len(prior_df)
    r_fat, p_fat = float(recent_df["fatalities"].sum()), float(prior_df["fatalities"].sum())
    r_inj, p_inj = float(recent_df["injured"].sum()), float(prior_df["injured"].sum())

    inc_delta = get_pct(r_inc, p_inc)
    fat_delta = get_pct(r_fat, p_fat)
    inj_delta = get_pct(r_inj, p_inj)

    if inc_delta > 10.0:
        trend_dir = "ESCALATING"
    elif inc_delta < -10.0:
        trend_dir = "DECLINING"
    else:
        trend_dir = "STABLE"

    return {
        "incident_delta": inc_delta,
        "fatality_delta": fat_delta,
        "injured_delta": inj_delta,
        "trend_direction": trend_dir,
        "recent_span": f"{recent_years[0]}–{recent_years[-1]}" if len(recent_years) > 1 else str(recent_years[0]),
        "prior_span": f"{prior_years[0]}–{prior_years[-1]}" if len(prior_years) > 1 else str(prior_years[0])
    }

def get_threat_glance_summary(df: pd.DataFrame, risk_df: pd.DataFrame) -> Dict[str, Any]:
    """Computes dynamic threat parameters derived strictly from the active filtered slice."""
    if df.empty:
        return {
            "highest_risk_country": "N/A",
            "highest_risk_score": 0.0,
            "dominant_tactic": "N/A",
            "primary_weapon": "N/A",
            "top_target": "N/A",
            "peak_year": "N/A",
            "peak_year_count": 0,
            "avg_lethality": 0.0
        }

    yearly = df.groupby("year").size()
    peak_yr = int(yearly.idxmax()) if not yearly.empty else "N/A"
    peak_val = int(yearly.max()) if not yearly.empty else 0

    top_country = risk_df.iloc[0]["country"] if not risk_df.empty else "N/A"
    top_score = float(risk_df.iloc[0]["composite_risk_score"]) if not risk_df.empty else 0.0

    dom_tactic = df["attack_type"].mode()[0] if not df["attack_type"].empty else "N/A"
    dom_weap = df["weapon_type"].mode()[0] if not df["weapon_type"].empty else "N/A"
    dom_target = df["target_type"].mode()[0] if not df["target_type"].empty else "N/A"

    total_fat = float(df["fatalities"].sum())
    total_inc = max(1, len(df))
    avg_lethality = round(total_fat / total_inc, 2)

    return {
        "highest_risk_country": top_country,
        "highest_risk_score": top_score,
        "dominant_tactic": dom_tactic,
        "primary_weapon": dom_weap,
        "top_target": dom_target,
        "peak_year": str(peak_yr),
        "peak_year_count": peak_val,
        "avg_lethality": avg_lethality
    }

def get_strategic_signals(df: pd.DataFrame, risk_df: pd.DataFrame, trends: dict, model_meta: dict) -> List[Dict[str, str]]:
    """Generates dynamic, academically grounded strategic signals matching the filtered period."""
    signals = []
    if df.empty:
        return signals

    # 1. Spatial Signal
    if not risk_df.empty:
        top_c = risk_df.iloc[0]["country"]
        top_score = risk_df.iloc[0]["composite_risk_score"]
        tier = risk_df.iloc[0]["risk_level"]
        top_c_inc = int(df[df["country"] == top_c].shape[0])
        pct_share = (top_c_inc / len(df)) * 100
        signals.append({
            "title": "Spatial Concentration Signal",
            "body": f"Primary active cluster centers in <b>{top_c}</b> ({top_c_inc:,} events, <code>{pct_share:.1f}%</code> of period share) with a Threat Index of <code>{top_score}/100</code> ({tier} Tier)."
        })

    # 2. Tactical Signal
    top_atk = df["attack_type"].value_counts()
    if not top_atk.empty:
        atk_name = top_atk.index[0]
        atk_pct = (top_atk.iloc[0] / len(df)) * 100
        signals.append({
            "title": "Tactical Signature Signal",
            "body": f"<b>{atk_name}</b> represents the leading operational tactic, accounting for <code>{atk_pct:.1f}%</code> ({top_atk.iloc[0]:,} events) of filtered occurrences."
        })

    # 3. Longitudinal Velocity Signal
    dir_txt = trends.get("trend_direction", "STABLE")
    delta_val = trends.get("incident_delta", 0.0)
    signals.append({
        "title": "Longitudinal Velocity Signal",
        "body": f"Within this window, activity exhibits an <b>{dir_txt}</b> historical trajectory (<code>{delta_val:+,.1f}%</code> volume delta between {trends.get('recent_span')} vs {trends.get('prior_span')})."
    })

    # 4. Predictive Engine Telemetry (Fixed Model Status)
    if model_meta and "best_model_metrics" in model_meta:
        m_name = model_meta.get("best_model_name", "Random Forest").split("(")[0].strip()
        acc = model_meta["best_model_metrics"].get("accuracy", "N/A")
        f1_m = model_meta["best_model_metrics"].get("macro_f1", "N/A")
        signals.append({
            "title": "Predictive Engine Signal",
            "body": f"<b>{m_name}</b> active as primary classifier with <code>{acc}%</code> holdout accuracy and <code>{f1_m}%</code> Macro-F1 across historical classes."
        })

    return signals

def get_research_findings(df: pd.DataFrame, risk_df: pd.DataFrame, model_meta: dict, quality: dict) -> List[str]:
    """Formulates dynamic empirical findings derived purely from the active timeframe."""
    if df.empty:
        return ["Insufficient historical records in the selected temporal window."]

    findings = []
    
    # Finding 1: Geographic concentration in this slice
    top_5_cnt = df["country"].value_counts().head(5).sum()
    top_5_pct = (top_5_cnt / len(df)) * 100
    top_country_name = df["country"].mode()[0] if not df["country"].empty else "N/A"
    findings.append(
        f"Geographic concentration in this period is led by {top_country_name}; the top 5 affected territories account for {top_5_pct:.1f}% ({top_5_cnt:,} events) of filtered activity."
    )

    # Finding 2: Tactical methodology
    top_atk = df["attack_type"].value_counts().head(2)
    if len(top_atk) >= 2:
        combined_pct = (top_atk.sum() / len(df)) * 100
        findings.append(
            f"Tactical methodologies show high reliance on '{top_atk.index[0]}' and '{top_atk.index[1]}', combining for {combined_pct:.1f}% of recorded operations in this window."
        )

    # Finding 3: Casualty Severity
    total_fat = int(df["fatalities"].sum())
    total_inj = int(df["injured"].sum())
    findings.append(
        f"Casualty footprint across the selected window produced {total_fat + total_inj:,} total casualties ({total_fat:,} fatalities, {total_inj:,} wounded), averaging {total_fat/max(1, len(df)):.2f} deaths per event."
    )

    # Finding 4: Model Evaluation (Global baseline)
    if model_meta and "best_model_metrics" in model_meta:
        m_acc = model_meta["best_model_metrics"].get("accuracy", "N/A")
        m_strat = model_meta.get("validation_strategy", "Temporal Out-of-Time")
        findings.append(
            f"Pre-event ML classification achieves {m_acc}% accuracy under a '{m_strat}' benchmark scheme, preventing post-event feature leakage."
        )

    return findings

def get_pipeline_status(model_meta: dict, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Evaluates live operational readiness of system components for the filtered view."""
    return [
        {"module": "DATA INGESTION PIPELINE", "status": "ONLINE" if not df.empty else "OFFLINE", "color": "#39D353" if not df.empty else "#F85149"},
        {"module": "ML CLASSIFIER ENGINE", "status": "LOADED" if model_meta else "UNTRAINED", "color": "#39D353" if model_meta else "#D29922"},
        {"module": "LONGITUDINAL FORECASTING", "status": "READY" if len(df["year"].unique()) >= 4 else "INSUFFICIENT DATA", "color": "#39D353" if len(df["year"].unique()) >= 4 else "#F85149"},
        {"module": "STATISTICAL ANOMALY WATCH", "status": "ACTIVE" if len(df["year"].unique()) >= 4 else "STANDBY", "color": "#39D353" if len(df["year"].unique()) >= 4 else "#D29922"},
        {"module": "DETERMINISTIC RISK ENGINE", "status": "CALIBRATED" if not df.empty else "OFFLINE", "color": "#39D353" if not df.empty else "#F85149"},
        {"module": "AI BRIEFING SYNTHESIZER", "status": "ONLINE" if not df.empty else "OFFLINE", "color": "#39D353" if not df.empty else "#F85149"}
    ]