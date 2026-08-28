"""
Deterministic Threat Risk Engine & Scenario Simulator
Location: ./src/risk_engine.py
Purpose: Computes transparent, globally normalized 0-100 composite risk scores,
         Shannon entropy diversity metrics, longitudinal risk trajectories, 
         sensitivity matrices, and scenario simulations.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Transparent Factor Weights
WEIGHT_FREQUENCY = 0.35
WEIGHT_FATALITY = 0.25
WEIGHT_INJURY = 0.15
WEIGHT_VELOCITY = 0.15
WEIGHT_DIVERSITY = 0.10

WEIGHTS_MAP = {
    "Incident Frequency": WEIGHT_FREQUENCY,
    "Fatality Severity": WEIGHT_FATALITY,
    "Injury Burden": WEIGHT_INJURY,
    "Recent Velocity": WEIGHT_VELOCITY,
    "Tactical Diversity": WEIGHT_DIVERSITY
}

def categorize_risk_tier(score: float) -> str:
    """Categorizes composite score into standardized risk classification tiers."""
    if score >= 80.0:
        return "Critical"
    elif score >= 60.0:
        return "High"
    elif score >= 40.0:
        return "Moderate"
    elif score >= 20.0:
        return "Low"
    return "Minimal"

def _compute_normalized_entropy(series: pd.Series, total_possible_classes: int) -> float:
    """
    Computes Normalized Shannon Entropy: H = -sum(p * ln(p)) / ln(K)
    Measures tactical operational balance rather than raw existence of classes.
    """
    if series.empty or total_possible_classes <= 1:
        return 0.0
    counts = series.value_counts().values
    probs = counts / float(counts.sum())
    # Shannon Entropy
    h = -np.sum(probs * np.log(probs + 1e-12))
    # Normalized by maximum possible entropy for K classes
    max_h = np.log(total_possible_classes)
    norm_h = float(np.clip(h / max_h, 0.0, 1.0))
    return norm_h

def compute_country_risk_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes global sovereign risk index ranking across 100% of matching dataset records.
    Normalizes every component relative to the global maximum across all sovereign entities.
    """
    if df.empty:
        return pd.DataFrame()

    max_yr = int(df["year"].max())
    recent_threshold = max_yr - 3
    global_total_tactics = max(2, int(df["attack_type"].nunique()))

    records = []
    for country, group in df.groupby("country"):
        incidents = len(group)
        fatalities = float(group["fatalities"].sum())
        injuries = float(group["injured"].sum())
        recent_incidents = len(group[group["year"] >= recent_threshold])
        raw_diversity_count = int(group["attack_type"].nunique())
        
        # Calculate Normalized Shannon Entropy
        entropy_val = _compute_normalized_entropy(group["attack_type"], global_total_tactics)

        records.append({
            "country": country,
            "region": group["region"].iloc[0] if "region" in group.columns else "Unknown",
            "total_incidents": incidents,
            "total_fatalities": fatalities,
            "total_injured": injuries,
            "recent_incidents": recent_incidents,
            "attack_diversity_count": raw_diversity_count,
            "tactical_entropy": entropy_val
        })

    risk_df = pd.DataFrame(records)

    # Global maximums across all sovereign entities
    max_freq = max(1.0, float(risk_df["total_incidents"].max()))
    max_fat = max(1.0, float(risk_df["total_fatalities"].max()))
    max_inj = max(1.0, float(risk_df["total_injured"].max()))
    max_vel = max(1.0, float(risk_df["recent_incidents"].max()))
    max_entropy = max(1e-5, float(risk_df["tactical_entropy"].max()))

    # Global Log1p Normalization relative to global maximums
    risk_df["norm_freq"] = ((np.log1p(risk_df["total_incidents"]) / np.log1p(max_freq)) * 100.0).clip(0.0, 100.0).round(1)
    risk_df["norm_fatality"] = ((np.log1p(risk_df["total_fatalities"]) / np.log1p(max_fat)) * 100.0).clip(0.0, 100.0).round(1)
    risk_df["norm_injury"] = ((np.log1p(risk_df["total_injured"]) / np.log1p(max_inj)) * 100.0).clip(0.0, 100.0).round(1)
    risk_df["norm_velocity"] = ((np.log1p(risk_df["recent_incidents"]) / np.log1p(max_vel)) * 100.0).clip(0.0, 100.0).round(1)
    
    # Entropy-based Tactical Diversity: 0–100 scaled by maximum observed entropy
    risk_df["norm_diversity"] = ((risk_df["tactical_entropy"] / max_entropy) * 100.0).clip(0.0, 100.0).round(1)

    # Calculate exact composite formula
    risk_df["composite_risk_score"] = (
        WEIGHT_FREQUENCY * risk_df["norm_freq"] +
        WEIGHT_FATALITY * risk_df["norm_fatality"] +
        WEIGHT_INJURY * risk_df["norm_injury"] +
        WEIGHT_VELOCITY * risk_df["norm_velocity"] +
        WEIGHT_DIVERSITY * risk_df["norm_diversity"]
    ).round(1)

    risk_df["risk_level"] = risk_df["composite_risk_score"].apply(categorize_risk_tier)
    return risk_df.sort_values(by="composite_risk_score", ascending=False).reset_index(drop=True)

def decompose_risk_score(row: pd.Series) -> pd.DataFrame:
    """Deconstructs composite threat score into raw metric, weight, and contribution."""
    components = [
        {
            "Risk Component Factor": "Incident Frequency",
            "Raw Observed Metric": f"{int(row.get('total_incidents', 0)):,} incidents",
            "Normalized Score (0-100)": float(row.get("norm_freq", 0.0)),
            "Weight": WEIGHT_FREQUENCY,
            "Contribution": round(float(row.get("norm_freq", 0.0)) * WEIGHT_FREQUENCY, 2)
        },
        {
            "Risk Component Factor": "Fatality Severity",
            "Raw Observed Metric": f"{int(row.get('total_fatalities', 0)):,} fatalities",
            "Normalized Score (0-100)": float(row.get("norm_fatality", 0.0)),
            "Weight": WEIGHT_FATALITY,
            "Contribution": round(float(row.get("norm_fatality", 0.0)) * WEIGHT_FATALITY, 2)
        },
        {
            "Risk Component Factor": "Injury Burden",
            "Raw Observed Metric": f"{int(row.get('total_injured', 0)):,} non-fatal casualties",
            "Normalized Score (0-100)": float(row.get("norm_injury", 0.0)),
            "Weight": WEIGHT_INJURY,
            "Contribution": round(float(row.get("norm_injury", 0.0)) * WEIGHT_INJURY, 2)
        },
        {
            "Risk Component Factor": "Recent Operational Velocity",
            "Raw Observed Metric": f"{int(row.get('recent_incidents', 0)):,} recent era events",
            "Normalized Score (0-100)": float(row.get("norm_velocity", 0.0)),
            "Weight": WEIGHT_VELOCITY,
            "Contribution": round(float(row.get("norm_velocity", 0.0)) * WEIGHT_VELOCITY, 2)
        },
        {
            "Risk Component Factor": "Tactical Diversity (Entropy)",
            "Raw Observed Metric": f"{int(row.get('attack_diversity_count', 0))} tactics (H={float(row.get('tactical_entropy', 0.0)):.2f})",
            "Normalized Score (0-100)": float(row.get("norm_diversity", 0.0)),
            "Weight": WEIGHT_DIVERSITY,
            "Contribution": round(float(row.get("norm_diversity", 0.0)) * WEIGHT_DIVERSITY, 2)
        }
    ]
    return pd.DataFrame(components)

def compute_country_risk_trajectory(df: pd.DataFrame, country_name: str) -> Dict[str, Any]:
    df_c = df[df["country"] == country_name].copy()
    if df_c.empty:
        return {"error": f"No data found for country: {country_name}"}

    years = sorted(df_c["year"].unique())
    max_annual_inc = max(1.0, float(df.groupby(["country", "year"]).size().max()))
    max_annual_fat = max(1.0, float(df.groupby(["country", "year"])["fatalities"].sum().max()))
    max_annual_inj = max(1.0, float(df.groupby(["country", "year"])["injured"].sum().max()))
    max_annual_div = max(2, int(df["attack_type"].nunique()))

    timeline = []
    prev_tier = None
    transitions = []

    for yr in years:
        grp = df_c[df_c["year"] == yr]
        inc = len(grp)
        fat = float(grp["fatalities"].sum())
        inj = float(grp["injured"].sum())
        entropy_val = _compute_normalized_entropy(grp["attack_type"], max_annual_div)

        n_freq = (np.log1p(inc) / np.log1p(max_annual_inc)) * 100.0
        n_fat = (np.log1p(fat) / np.log1p(max_annual_fat)) * 100.0
        n_inj = (np.log1p(inj) / np.log1p(max_annual_inj)) * 100.0
        n_div = entropy_val * 100.0

        prior_grp = df_c[(df_c["year"] >= yr - 2) & (df_c["year"] < yr)]
        prior_avg = len(prior_grp) / 2.0 if not prior_grp.empty else 0.0
        n_vel = (np.log1p(inc) / np.log1p(max(inc, prior_avg * 2.0, 1.0))) * 100.0 if inc > 0 else 0.0

        score = round(
            WEIGHT_FREQUENCY * min(100.0, n_freq) +
            WEIGHT_FATALITY * min(100.0, n_fat) +
            WEIGHT_INJURY * min(100.0, n_inj) +
            WEIGHT_VELOCITY * min(100.0, n_vel) +
            WEIGHT_DIVERSITY * min(100.0, n_div),
            1
        )
        tier = categorize_risk_tier(score)

        if prev_tier is not None and tier != prev_tier:
            transitions.append({
                "Year": yr,
                "Previous Tier": prev_tier,
                "New Tier": tier,
                "Score Delta": round(score - timeline[-1]["Threat Index"], 1) if timeline else 0.0,
                "Transition Type": f"{prev_tier} → {tier}"
            })

        timeline.append({
            "Year": yr,
            "Threat Index": score,
            "Risk Tier": tier,
            "Incidents": inc,
            "Fatalities": int(fat),
            "Injuries": int(inj)
        })
        prev_tier = tier

    df_timeline = pd.DataFrame(timeline)
    
    if len(df_timeline) >= 3:
        first_half = df_timeline.head(len(df_timeline) // 2)["Threat Index"].mean()
        second_half = df_timeline.tail(len(df_timeline) // 2)["Threat Index"].mean()
        diff = second_half - first_half
        std_dev = df_timeline["Threat Index"].std()

        if std_dev > 18.0:
            classification = "Volatile / High Fluctuation"
        elif diff >= 8.0:
            classification = "Rising / Escalating"
        elif diff <= -8.0:
            classification = "Declining / De-escalating"
        else:
            classification = "Stable / Range-Bound"
    else:
        classification = "Stable / Insufficient Longitudinal Points"

    peak_idx = df_timeline["Threat Index"].idxmax()
    min_idx = df_timeline["Threat Index"].idxmin()

    return {
        "timeline": df_timeline,
        "transitions": pd.DataFrame(transitions),
        "trajectory_classification": classification,
        "peak_year": int(df_timeline.loc[peak_idx, "Year"]),
        "peak_score": float(df_timeline.loc[peak_idx, "Threat Index"]),
        "min_year": int(df_timeline.loc[min_idx, "Year"]),
        "min_score": float(df_timeline.loc[min_idx, "Threat Index"]),
        "overall_delta": round(float(df_timeline.iloc[-1]["Threat Index"] - df_timeline.iloc[0]["Threat Index"]), 1)
    }

def compute_risk_sensitivity(norm_freq: float, norm_fat: float, norm_inj: float, norm_vel: float, norm_div: float, perturbation_pct: float = 10.0) -> Dict[str, Any]:
    baseline_score = round(
        WEIGHT_FREQUENCY * norm_freq +
        WEIGHT_FATALITY * norm_fat +
        WEIGHT_INJURY * norm_inj +
        WEIGHT_VELOCITY * norm_vel +
        WEIGHT_DIVERSITY * norm_div,
        2
    )

    factors = [
        ("Incident Frequency", norm_freq, WEIGHT_FREQUENCY),
        ("Fatality Severity", norm_fat, WEIGHT_FATALITY),
        ("Injury Burden", norm_inj, WEIGHT_INJURY),
        ("Recent Operational Velocity", norm_vel, WEIGHT_VELOCITY),
        ("Tactical Diversity", norm_div, WEIGHT_DIVERSITY)
    ]

    sensitivity_rows = []
    for name, val, weight in factors:
        perturbed_val = min(100.0, val * (1.0 + perturbation_pct / 100.0))
        delta_val = perturbed_val - val
        new_score = baseline_score + (delta_val * weight)
        abs_delta = round(new_score - baseline_score, 2)
        pct_delta = round((abs_delta / max(0.01, baseline_score)) * 100.0, 2)

        sensitivity_rows.append({
            "Risk Component": name,
            "Baseline Value": round(val, 1),
            f"Perturbed (+{int(perturbation_pct)}%)": round(perturbed_val, 1),
            "Factor Weight": weight,
            "New Threat Index": round(new_score, 1),
            "Absolute Delta (Pts)": abs_delta,
            "Sensitivity Shift (%)": pct_delta
        })

    df_sens = pd.DataFrame(sensitivity_rows).sort_values("Absolute Delta (Pts)", ascending=False).reset_index(drop=True)
    return {
        "sensitivity_table": df_sens,
        "most_sensitive_component": df_sens.iloc[0]["Risk Component"],
        "baseline_score": baseline_score
    }

def simulate_threat_score(freq: float, fatality: float, injury: float, velocity: float, diversity: float) -> dict:
    score = round(
        WEIGHT_FREQUENCY * freq +
        WEIGHT_FATALITY * fatality +
        WEIGHT_INJURY * injury +
        WEIGHT_VELOCITY * velocity +
        WEIGHT_DIVERSITY * diversity,
        1
    )
    score_bounded = min(100.0, max(0.0, score))
    return {
        "simulated_score": score_bounded,
        "simulated_tier": categorize_risk_tier(score_bounded),
        "weights": {
            "Frequency Contribution (35%)": round(freq * WEIGHT_FREQUENCY, 2),
            "Fatality Severity (25%)": round(fatality * WEIGHT_FATALITY, 2),
            "Injury Burden (15%)": round(injury * WEIGHT_INJURY, 2),
            "Recent Velocity (15%)": round(velocity * WEIGHT_VELOCITY, 2),
            "Tactic Diversity (10%)": round(diversity * WEIGHT_DIVERSITY, 2)
        }
    }

def compute_evidence_reliability(df_c: pd.DataFrame) -> Dict[str, Any]:
    if df_c.empty:
        return {"total_records": 0, "reliability_score": 0.0, "reliability_tier": "No Data"}

    total_records = len(df_c)
    geo_valid = int((df_c["latitude"].notnull() & df_c["longitude"].notnull()).sum())
    geo_rate = round((geo_valid / max(1, total_records)) * 100.0, 1)

    cas_valid = int((df_c["fatalities"].notnull() & df_c["injured"].notnull()).sum())
    cas_rate = round((cas_valid / max(1, total_records)) * 100.0, 1)

    yr_span = int(df_c["year"].max() - df_c["year"].min() + 1) if not df_c.empty else 0
    volume_score = min(100.0, (np.log1p(total_records) / np.log1p(2500)) * 100.0)
    reliability_index = round(0.40 * volume_score + 0.30 * geo_rate + 0.30 * cas_rate, 1)

    if reliability_index >= 80.0:
        rel_tier, rel_color = "High Evidence Depth", "#39D353"
    elif reliability_index >= 50.0:
        rel_tier, rel_color = "Moderate Evidence Depth", "#58A6FF"
    else:
        rel_tier, rel_color = "Limited Evidence Depth", "#FFA657"

    return {
        "total_records": total_records,
        "geocoding_rate": geo_rate,
        "casualty_reporting_rate": cas_rate,
        "temporal_span_years": yr_span,
        "reliability_score": reliability_index,
        "reliability_tier": rel_tier,
        "reliability_color": rel_color
    }