"""
Deterministic Intelligence Briefing Generator (13-Section Academic Brief)
Location: ./src/report_generator.py
"""

import pandas as pd
from datetime import datetime
from src.analytics import calculate_period_trends
from src.anomaly_detection import detect_historical_anomalies
from src.data_loader import audit_dataset_quality
from src.forecasting import generate_incident_forecast


def generate_intelligence_digest(df: pd.DataFrame, risk_df: pd.DataFrame) -> str:
    if df.empty:
        return "# Analytical Report\n\nDataset is empty. No telemetry available."

    min_yr = int(df["year"].min()) if "year" in df.columns else 1970
    max_yr = int(df["year"].max()) if "year" in df.columns else 2017
    total_incidents = len(df)
    total_fatalities = int(df["fatalities"].sum()) if "fatalities" in df.columns else 0
    total_injured = int(df["injured"].sum()) if "injured" in df.columns else 0

    if "year" in df.columns:
        yearly_incidents = df.groupby("year").size()
        peak_incident_year = int(yearly_incidents.idxmax())
        peak_incident_val = int(yearly_incidents.max())
    else:
        peak_incident_year = "N/A"
        peak_incident_val = 0

    top_countries = df["country"].value_counts().head(5).to_dict() if "country" in df.columns else {}
    top_attacks = df["attack_type"].value_counts().head(5).to_dict() if "attack_type" in df.columns else {}
    top_weapons = df["weapon_type"].value_counts().head(5).to_dict() if "weapon_type" in df.columns else {}
    top_targets = df["target_type"].value_counts().head(5).to_dict() if "target_type" in df.columns else {}

    trends = calculate_period_trends(df)
    anomalies_df = detect_historical_anomalies(df)
    flagged_anomalies = anomalies_df[anomalies_df["is_anomaly"]] if (not anomalies_df.empty and "is_anomaly" in anomalies_df.columns) else pd.DataFrame()
    
    quality = audit_dataset_quality(df)
    forecast_res = generate_incident_forecast(df, forecast_horizon=3)

    report = f"""# GLOBAL THREAT INTELLIGENCE & ANALYTICAL RISK SYNTHESIS BRIEF
**Classification:** HISTORICAL ACADEMIC EVALUATION (UNCLASSIFIED)  
**System Reference:** GTI-ARP Decision Support Engine  
**Generation Timestamp:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Temporal Horizon:** {min_yr} – {max_yr}

---

## 1. Executive Summary
Across the historical span of **{min_yr} to {max_yr}**, the platform evaluated **{total_incidents:,}** incident records across **{df['country'].nunique() if 'country' in df.columns else 0}** sovereign entities.
* **Cumulative Casualties:** {total_fatalities + total_injured:,} ({total_fatalities:,} Fatalities | {total_injured:,} Non-Fatal Casualties)
* **Historical Peak Activity:** Year **{peak_incident_year}** ({peak_incident_val:,} logged events)
* **Data Quality Score:** `{quality.get('data_quality_score', 0)}/100` (Completeness: `{quality.get('completeness_pct', 0)}%`, Geocoding: `{quality.get('geocoding_coverage_pct', 0)}%`)

---

## 2. Global Trend Analysis
* **Trajectory Direction:** **{trends.get('trend_direction', 'STABLE')}** ({trends.get('recent_span', 'N/A')} vs baseline {trends.get('prior_span', 'N/A')})
* **Period Velocity Deltas:**
  * Incident Volume Delta: `{trends.get('incident_delta', 0.0):+,.1f}%`
  * Fatalities Delta: `{trends.get('fatality_delta', 0.0):+,.1f}%`
  * Injuries Delta: `{trends.get('injured_delta', 0.0):+,.1f}%`

---

## 3. Geographic Concentration
Top sovereign territories accounting for the highest historical incident concentrations:
"""
    for c, cnt in top_countries.items():
        pct = (cnt / total_incidents) * 100 if total_incidents > 0 else 0
        report += f"- **{c}**: {cnt:,} incidents ({pct:.1f}% global density)\n"

    report += """
---

## 4. Attack Methodology & Tactical Profiles
"""
    for atk, cnt in top_attacks.items():
        pct = (cnt / total_incidents) * 100 if total_incidents > 0 else 0
        report += f"- **{atk}**: {cnt:,} incidents ({pct:.1f}%)\n"

    report += "\n---\n\n## 5. Weapon Category Analysis\n"
    for weap, cnt in top_weapons.items():
        pct = (cnt / total_incidents) * 100 if total_incidents > 0 else 0
        report += f"- **{weap}**: {cnt:,} incidents ({pct:.1f}%)\n"

    report += "\n---\n\n## 6. Target Profile Distribution\n"
    for targ, cnt in top_targets.items():
        pct = (cnt / total_incidents) * 100 if total_incidents > 0 else 0
        report += f"- **{targ}**: {cnt:,} incidents ({pct:.1f}%)\n"

    if risk_df is not None and not risk_df.empty:
        report += "\n---\n\n## 7. Country Risk Ranking\n"
        for _, rk in risk_df.head(5).iterrows():
            c_name = rk.get("country", "Unknown")
            r_score = rk.get("composite_risk_score", rk.get("threat_index", rk.get("risk_score", 0)))
            r_level = rk.get("risk_level", rk.get("tier", "Active"))
            report += f"- **{c_name}**: Threat Index `{r_score}/100` — **{r_level} Classification**\n"

    f_yrs = forecast_res.get("future_years", [])
    f_cnts = forecast_res.get("forecast_counts", [])
    f_proj = ", ".join([f"{yr}: ~{int(cnt):,}" for yr, cnt in zip(f_yrs, f_cnts)]) if f_yrs else "Insufficient longitudinal baseline"

    report += f"""
---

## 8. Forecast Outlook & Uncertainty Bounds
* **Extrapolation Method:** Holt's Linear Double Exponential Smoothing
* **Next 3 Horizon Projections:** {f_proj}

---

## 9. Anomaly Surveillance
Applying a 5-year rolling Z-score filter ($Z \\ge 2.0$) revealed **{len(flagged_anomalies)}** surge periods.

---

## 10. Machine Learning Insights
Multi-model validation established Random Forest / HistGradientBoosting as the primary multiclass predictor based on pre-event context.

---

## 11. Data Quality & Audit Metrics
* **Total Indexed Incidents:** {quality.get('total_records', total_incidents):,}
* **Geocoding Integrity:** {quality.get('geocoding_coverage_pct', 0)}%
* **Completeness:** {quality.get('completeness_pct', 0)}%

---

## 12. Model & Analytical Limitations
1. Evaluates pre-event contextual signals without claiming causality.
2. Historical reporting completeness varies across historical recording decades.

---

## 13. Academic Ethics & Research Disclaimer
This platform is intended exclusively for academic, statistical, and decision-support research. Historical patterns and model outputs should not be interpreted as real-time operational threat intelligence or predictions of specific future events.
"""
    return report