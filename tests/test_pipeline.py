"""
Automated Unit & Integration Test Suite
Location: ./tests/test_pipeline.py
Execution: pytest tests/
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_loader import generate_demo_dataset, standardize_dataframe, audit_dataset_quality
from src.risk_engine import (
    compute_country_risk_index,
    decompose_risk_score,
    compute_country_risk_trajectory,
    compute_risk_sensitivity,
    simulate_threat_score,
    compute_evidence_reliability
)
from src.forecasting import (
    generate_incident_forecast,
    run_forecasting_models,
    run_walk_forward_validation,
    _compute_metrics,
    _fit_predict_model
)
from src.anomaly_detection import detect_historical_anomalies
from src.preprocessing import prepare_data_for_training, build_preprocessor

def test_demo_dataset_generation():
    df = generate_demo_dataset()
    assert not df.empty
    assert len(df) >= 1000
    assert "year" in df.columns
    assert "fatalities" in df.columns

def test_standardize_dataframe():
    raw_df = pd.DataFrame({
        "iyear": [2000, 2001],
        "country_txt": ["Iraq", "India"],
        "nkill": [2, 0],
        "attacktype1_txt": ["Bombing", "Armed Assault"]
    })
    clean_df = standardize_dataframe(raw_df)
    assert "year" in clean_df.columns
    assert "country" in clean_df.columns
    assert "fatalities" in clean_df.columns
    assert "attack_type" in clean_df.columns

def test_data_quality_audit():
    df = generate_demo_dataset()
    quality = audit_dataset_quality(df)
    assert "data_quality_score" in quality
    assert 0.0 <= quality["data_quality_score"] <= 100.0

def test_risk_engine_calculation():
    df = generate_demo_dataset()
    risk_df = compute_country_risk_index(df)
    assert not risk_df.empty
    assert "composite_risk_score" in risk_df.columns
    assert risk_df["composite_risk_score"].between(0.0, 100.0).all()

def test_risk_score_decomposition():
    df = generate_demo_dataset()
    risk_df = compute_country_risk_index(df)
    row = risk_df.iloc[0]
    decomp = decompose_risk_score(row)
    assert len(decomp) == 5
    assert "Contribution (Points)" in decomp.columns
    total_contrib = decomp["Contribution (Points)"].sum()
    assert abs(total_contrib - row["composite_risk_score"]) < 1.0

def test_risk_trajectory():
    df = generate_demo_dataset()
    country = df["country"].iloc[0]
    traj = compute_country_risk_trajectory(df, country)
    assert "timeline" in traj
    assert not traj["timeline"].empty
    assert "trajectory_classification" in traj
    assert "transitions" in traj

def test_risk_sensitivity():
    sens = compute_risk_sensitivity(norm_freq=80.0, norm_fat=70.0, norm_inj=50.0, norm_vel=60.0, norm_div=40.0)
    assert "sensitivity_table" in sens
    assert "most_sensitive_component" in sens
    assert len(sens["sensitivity_table"]) == 5

def test_evidence_reliability():
    df = generate_demo_dataset()
    country = df["country"].iloc[0]
    rel = compute_evidence_reliability(df[df["country"] == country])
    assert "reliability_score" in rel
    assert 0.0 <= rel["reliability_score"] <= 100.0
    assert "reliability_tier" in rel

def test_risk_scenario_simulator():
    sim = simulate_threat_score(freq=80, fatality=70, injury=50, velocity=90, diversity=60)
    assert "simulated_score" in sim
    assert sim["simulated_tier"] in ["Critical", "High", "Moderate", "Low", "Minimal"]

# ==========================================
# FORECASTING UNIT TESTS
# ==========================================

def test_forecasting_metrics_division_by_zero():
    actuals = np.array([0.0, 0.0, 10.0, 50.0])
    preds = np.array([2.0, 0.0, 12.0, 45.0])
    mets = _compute_metrics(actuals, preds)
    assert not np.isnan(mets["MAPE (%)"])
    assert not np.isnan(mets["sMAPE (%)"])
    assert mets["MAE (Events)"] >= 0.0

def test_forecasting_leakage_and_horizon():
    counts = np.array([100, 120, 130, 150, 180, 210, 240, 260, 290, 310, 350, 400], dtype=float)
    res = run_forecasting_models(counts, test_periods=3)
    assert "leaderboard" in res
    assert len(res["leaderboard"]) == 4
    assert res["holdout_periods"] == 3
    # Check that predictions exist for holdout
    for m, p in res["backtest_predictions"].items():
        assert len(p) == 3

def test_forecasting_model_chart_consistency():
    df = generate_demo_dataset()
    res = generate_incident_forecast(df, forecast_horizon=4)
    assert "forecast_counts" in res
    assert len(res["forecast_counts"]) == 4
    assert "active_model_name" in res
    # Active model must match best backtested model by default
    assert res["active_model_name"] == res["backtest_results"]["best_model_name"]
    assert len(res["lower_bound"]) == 4
    assert len(res["upper_bound"]) == 4

def test_walk_forward_validation():
    counts = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200], dtype=float)
    df_wf = run_walk_forward_validation(counts, min_train=10)
    assert not df_wf.empty
    assert "MAE (Events)" in df_wf.columns
    assert len(df_wf) == 4

def test_anomaly_detection():
    df = generate_demo_dataset()
    anom_df = detect_historical_anomalies(df, window=5, z_threshold=2.0)
    assert not anom_df.empty
    assert "is_anomaly" in anom_df.columns
    assert "z_score" in anom_df.columns