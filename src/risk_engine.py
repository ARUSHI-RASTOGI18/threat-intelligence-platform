"""
Deterministic Threat Risk Engine & Scenario Simulator
Location: ./src/risk_engine.py
"""

import pandas as pd
import numpy as np

WEIGHT_FREQUENCY = 0.35
WEIGHT_FATALITY = 0.30
WEIGHT_INJURY = 0.15
WEIGHT_VELOCITY = 0.10
WEIGHT_DIVERSITY = 0.10

def compute_country_risk_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    max_yr = df["year"].max()
    recent_threshold = max_yr - 3

    records = []
    for country, group in df.groupby("country"):
        incidents = len(group)
        fatalities = group["fatalities"].sum()
        injuries = group["injured"].sum()
        recent_incidents = len(group[group["year"] >= recent_threshold])
        diversity = group["attack_type"].nunique()

        records.append({
            "country": country,
            "region": group["region"].iloc[0],
            "total_incidents": incidents,
            "total_fatalities": fatalities,
            "total_injured": injuries,
            "recent_incidents": recent_incidents,
            "attack_diversity": diversity
        })

    risk_df = pd.DataFrame(records)

    def min_max_log(series: pd.Series) -> pd.Series:
        log_s = np.log1p(series)
        min_v, max_v = log_s.min(), log_s.max()
        if max_v == min_v:
            return pd.Series(0.0, index=series.index)
        return (log_s - min_v) / (max_v - min_v)

    def min_max_linear(series: pd.Series) -> pd.Series:
        min_v, max_v = series.min(), series.max()
        if max_v == min_v:
            return pd.Series(0.0, index=series.index)
        return (series - min_v) / (max_v - min_v)

    risk_df["norm_freq"] = (min_max_log(risk_df["total_incidents"]) * 100).round(1)
    risk_df["norm_fatality"] = (min_max_log(risk_df["total_fatalities"]) * 100).round(1)
    risk_df["norm_injury"] = (min_max_log(risk_df["total_injured"]) * 100).round(1)
    risk_df["norm_velocity"] = (min_max_log(risk_df["recent_incidents"]) * 100).round(1)
    risk_df["norm_diversity"] = (min_max_linear(risk_df["attack_diversity"]) * 100).round(1)

    risk_df["composite_risk_score"] = (
        WEIGHT_FREQUENCY * risk_df["norm_freq"] +
        WEIGHT_FATALITY * risk_df["norm_fatality"] +
        WEIGHT_INJURY * risk_df["norm_injury"] +
        WEIGHT_VELOCITY * risk_df["norm_velocity"] +
        WEIGHT_DIVERSITY * risk_df["norm_diversity"]
    ).round(1)

    def categorize_tier(score):
        if score >= 80.0:
            return "Critical"
        elif score >= 60.0:
            return "High"
        elif score >= 40.0:
            return "Moderate"
        elif score >= 20.0:
            return "Low"
        return "Minimal"

    risk_df["risk_level"] = risk_df["composite_risk_score"].apply(categorize_tier)
    return risk_df.sort_values(by="composite_risk_score", ascending=False).reset_index(drop=True)

def simulate_threat_score(freq: float, fatality: float, injury: float, velocity: float, diversity: float) -> dict:
    score = round(
        WEIGHT_FREQUENCY * freq +
        WEIGHT_FATALITY * fatality +
        WEIGHT_INJURY * injury +
        WEIGHT_VELOCITY * velocity +
        WEIGHT_DIVERSITY * diversity,
        1
    )
    
    if score >= 80.0:
        tier = "Critical"
    elif score >= 60.0:
        tier = "High"
    elif score >= 40.0:
        tier = "Moderate"
    elif score >= 20.0:
        tier = "Low"
    else:
        tier = "Minimal"

    return {
        "simulated_score": min(100.0, max(0.0, score)),
        "simulated_tier": tier,
        "weights": {
            "Frequency (35%)": freq * WEIGHT_FREQUENCY,
            "Fatality (30%)": fatality * WEIGHT_FATALITY,
            "Injury (15%)": injury * WEIGHT_INJURY,
            "Velocity (10%)": velocity * WEIGHT_VELOCITY,
            "Diversity (10%)": diversity * WEIGHT_DIVERSITY
        }
    }