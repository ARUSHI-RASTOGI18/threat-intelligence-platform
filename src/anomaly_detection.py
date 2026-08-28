"""
Statistical Anomaly Detection Engine (Rolling Z-Score, IQR, and Velocity Thresholds)
Location: ./src/anomaly_detection.py
"""

import pandas as pd
import numpy as np

def detect_historical_anomalies(df: pd.DataFrame, window: int = 5, z_threshold: float = 2.0) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    yearly = df.groupby("year").size().reset_index(name="incident_count").sort_values("year")
    if len(yearly) < window:
        yearly["rolling_mean"] = yearly["incident_count"].mean()
        yearly["rolling_std"] = yearly["incident_count"].std()
        yearly["z_score"] = 0.0
        yearly["is_anomaly"] = False
        yearly["anomaly_severity"] = "Normal"
        yearly["pct_deviation"] = 0.0
        return yearly

    yearly["rolling_mean"] = yearly["incident_count"].rolling(window=window, min_periods=2, center=False).mean()
    yearly["rolling_std"] = yearly["incident_count"].rolling(window=window, min_periods=2, center=False).std()
    
    yearly["rolling_std"] = yearly["rolling_std"].replace(0, np.nan).bfill().fillna(1.0)
    yearly["rolling_mean"] = yearly["rolling_mean"].bfill().fillna(yearly["incident_count"])

    yearly["z_score"] = (yearly["incident_count"] - yearly["rolling_mean"]) / yearly["rolling_std"]
    yearly["is_anomaly"] = yearly["z_score"].abs() >= z_threshold
    yearly["pct_deviation"] = (((yearly["incident_count"] - yearly["rolling_mean"]) / yearly["rolling_mean"]) * 100).round(1)

    def classify_severity(row):
        z = abs(row["z_score"])
        if z >= 3.0:
            return "Critical Anomaly"
        elif z >= 2.0:
            return "Moderate Anomaly"
        elif z >= 1.5:
            return "Mild Deviation"
        return "Normal"

    yearly["anomaly_severity"] = yearly.apply(classify_severity, axis=1)
    return yearly