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

    min_yr = int(df["year"].min())
    max_yr = int(df["year"].max())
    total_incidents = len(df)
    total_fatalities = int(df["fatalities"].sum())
    total_injured = int(df["injured"].sum())

    yearly_incidents = df.groupby("year").size()
    peak_incident_year = int(yearly_incidents.idxmax())
    peak_incident_val = int(yearly_incidents.max())

    top_countries = df["country"].value_counts().head(5).to_dict()
    top_attacks = df["attack_type"].value_counts().head(5).to_dict()
    top_weapons = df["weapon_type"].value_counts().head(5).to_dict()
    top_targets = df["target_type"].value_counts().head(5).to_dict()

    trends = calculate_period_trends(df)
    anomalies_df = detect_historical_anomalies(df)
    flagged_anomalies = anomalies_df[anomalies_df["is_anomaly"]] if not anomalies_df.empty else pd.DataFrame()
    
    quality = audit_dataset_quality(df)
    forecast_res = generate_incident_forecast(df, forecast_horizon=3)

    report = f"""# GLOBAL THREAT INTELLIGENCE & ANALYTICAL RISK SYNTHESIS BRIEF
**Classification:** HISTORICAL ACADEMIC EVALUATION (UNCLASSIFIED)  
**System Reference:** GTI-ARP Decision Support Engine  
**Generation Timestamp:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Temporal Horizon:** {min_yr} – {max_yr}

---

## 1. Executive Summary
Across the historical span of **{min_yr} to {max_yr}**, the platform evaluated **{total_incidents:,}** incident records across **{df['country'].nunique()}** sovereign entities.
* **Cumulative Casualties:** {total_fatalities + total_injured:,} ({total_fatalities:,} Fatalities | {total_injured:,} Non-Fatal Casualties)
* **Historical Peak Activity:** Year **{peak_incident_year}** ({peak_incident_val:,} logged events)
* **Data Quality Score:** `{quality['data_quality_score']}/100` (Completeness: `{quality['completeness_pct']}%`, Geocoding: `{quality['geocoding_coverage_pct']}%`)

---

## 2. Global Trend Analysis
* **Trajectory Direction:** **{trends['trend_direction']}** ({trends['recent_span']} vs baseline {trends['prior_span']})
* **Period Velocity Deltas:**
  * Incident Volume Delta: `{trends['incident_delta']:+,.1f}%`
  * Fatalities Delta: `{trends['fatality_delta']:+,.1f}%`
  * Injuries Delta: `{trends['injured_delta']:+,.1f}%`

---

## 3. Geographic Concentration
Top sovereign territories accounting for the highest historical incident concentrations:
"""
    for c, cnt in top_countries.items():
        pct = (cnt / total_incidents) * 100
        report += f"- **{c}**: {cnt:,} incidents ({pct:.1f}% global density)\n"

    report += """
---

## 4. Attack Methodology & Tactical Profiles
"""
    for atk, cnt in top_attacks.items():
        report += f"- **{atk}**: {cnt:,} incidents ({(cnt/total_incidents)*100:.1f}%)\n"

    report += "\n---\n\n## 5. Weapon Category Analysis\n"
    for weap, cnt in top_weapons.items():
        report += f"- **{weap}**: {cnt:,} incidents ({(cnt/total_incidents)*100:.1f}%)\n"

    report += "\n---\n\n## 6. Target Profile Distribution\n"
    for targ, cnt in top_targets.items():
        report += f"- **{targ}**: {cnt:,} incidents ({(cnt/total_incidents)*100:.1f}%)\n"

    if not risk_df.empty:
        report += "\n---\n\n## 7. Country Risk Ranking\n"
        for _, rk in risk_df.head(5).iterrows():
            report += f"- **{rk['country']}**: Threat Index `{rk['composite_risk_score']}/100` — **{rk['risk_level']} Classification**\n"

    report += f"""
---

## 8. Forecast Outlook & Uncertainty Bounds
* **Extrapolation Method:** Holt's Linear Double Exponential Smoothing
* **Next 3 Horizon Projections:** {', '.join([f'{yr}: ~{int(cnt):,}' for yr, cnt in zip(forecast_res.get('future_years', []), forecast_res.get('forecast_counts', []))])}

---

## 9. Anomaly Surveillance
Applying a 5-year rolling Z-score filter ($Z \\ge 2.0$) revealed **{len(flagged_anomalies)}** surge periods.

---

## 10. Machine Learning Insights
Multi-model validation established Random Forest / HistGradientBoosting as the primary multiclass predictor based on pre-event context.

---

## 11. Data Quality & Audit Metrics
* **Total Indexed Incidents:** {quality['total_records']:,}
* **Geocoding Integrity:** {quality['geocoding_coverage_pct']}%
* **Completeness:** {quality['completeness_pct']}%

---

## 12. Model & Analytical Limitations
1. Evaluates pre-event contextual signals without claiming causality.
2. Historical reporting completeness varies across historical recording decades.

---

## 13. Academic Ethics & Research Disclaimer
This platform is intended exclusively for academic, statistical, and decision-support research. Historical patterns and model outputs should not be interpreted as real-time operational threat intelligence or predictions of specific future events.
"""
    return report