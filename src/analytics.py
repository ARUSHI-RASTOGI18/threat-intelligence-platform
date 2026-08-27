"""
Advanced Statistical & Comparative Analytics Engine
Location: ./src/analytics.py
Purpose: Pure statistical calculations for Anomaly Detection (Z-Score),
         Longitudinal Trend Velocities, and Sovereign Head-to-Head Comparative Profiling.
"""

import pandas as pd
import numpy as np

def detect_historical_anomalies(df: pd.DataFrame, window: int = 5, z_threshold: float = 2.0) -> pd.DataFrame:
    """
    Detects unusual surges in annual incident frequency using a rolling Z-score.
    Returns yearly aggregated stats with rolling mean, std, z-scores, and boolean anomaly flags.
    """
    if df.empty:
        return pd.DataFrame()

    yearly = df.groupby("year").size().reset_index(name="incident_count").sort_values("year")
    if len(yearly) < window:
        yearly["rolling_mean"] = yearly["incident_count"].mean()
        yearly["rolling_std"] = yearly["incident_count"].std()
        yearly["z_score"] = 0.0
        yearly["is_anomaly"] = False
        yearly["pct_deviation"] = 0.0
        return yearly

    yearly["rolling_mean"] = yearly["incident_count"].rolling(window=window, min_periods=2, center=False).mean()
    yearly["rolling_std"] = yearly["incident_count"].rolling(window=window, min_periods=2, center=False).std()
    
    # Handle zero or null standard deviation
    yearly["rolling_std"] = yearly["rolling_std"].replace(0, np.nan).bfill().fillna(1.0)
    yearly["rolling_mean"] = yearly["rolling_mean"].bfill().fillna(yearly["incident_count"])

    yearly["z_score"] = (yearly["incident_count"] - yearly["rolling_mean"]) / yearly["rolling_std"]
    yearly["is_anomaly"] = yearly["z_score"].abs() >= z_threshold
    yearly["pct_deviation"] = (((yearly["incident_count"] - yearly["rolling_mean"]) / yearly["rolling_mean"]) * 100).round(1)

    return yearly

def calculate_period_trends(df: pd.DataFrame) -> dict:
    """
    Computes percentage deltas across the most recent historical dataset period
    relative to the immediate preceding equivalent period.
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

    # Split available years into two recent equal spans (up to 3 years each)
    span_len = min(3, len(years) // 2) if len(years) >= 4 else 1
    recent_years = years[-span_len:]
    prior_years = years[-2 * span_len : -span_len]

    recent_df = df[df["year"].isin(recent_years)]
    prior_df = df[df["year"].isin(prior_years)]

    def get_pct(curr, prev):
        if prev == 0:
            return 100.0 if curr > 0 else 0.0
        return round(((curr - prev) / prev) * 100.0, 1)

    r_inc, p_inc = len(recent_df), len(prior_df)
    r_fat, p_fat = recent_df["fatalities"].sum(), prior_df["fatalities"].sum()
    r_inj, p_inj = recent_df["injured"].sum(), prior_df["injured"].sum()

    inc_delta = get_pct(r_inc, p_inc)
    fat_delta = get_pct(r_fat, p_fat)
    inj_delta = get_pct(r_inj, p_inj)

    if inc_delta > 10.0:
        trend_dir = "INCREASING"
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

def generate_risk_driver_explanation(row: pd.Series) -> list:
    """
    Generates deterministic, transparent bullet points explaining why
    a specific territory produced its analytical threat score.
    """
    drivers = []
    
    # Frequency
    if row.get("norm_freq", 0) >= 70:
        drivers.append("High historical incident frequency relative to global distribution.")
    elif row.get("norm_freq", 0) >= 35:
        drivers.append("Moderate aggregate incident volume observed across dataset.")
    else:
        drivers.append("Low baseline incident frequency.")

    # Lethality
    if row.get("norm_fatality", 0) >= 70:
        drivers.append("Substantial cumulative fatality count indicating severe event lethality.")
    elif row.get("norm_fatality", 0) >= 35:
        drivers.append("Moderate fatality density registered across recorded incidents.")

    # Velocity
    if row.get("norm_velocity", 0) >= 60:
        drivers.append("Elevated incident density during the most recent active dataset years.")
    else:
        drivers.append("Subdued or decelerating activity volume in later periods.")

    # Target Diversity
    if row.get("norm_diversity", 0) >= 70:
        drivers.append("Broad tactic diversity spanning multiple simultaneous attack methodologies.")
    else:
        drivers.append("Tactic concentration restricted to isolated attack categories.")

    return drivers