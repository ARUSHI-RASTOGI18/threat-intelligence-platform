"""
Module 6: Longitudinal Forecasting & Empirical Backtesting
Location: ./pages/5_Forecasting.py
"""

import streamlit as st
import plotly.graph_objects as go
from src.data_loader import load_analytical_data
from src.forecasting import generate_incident_forecast

st.set_page_config(page_title="Forecasting | GTI-ARP", page_icon="📈", layout="wide")

st.title("Longitudinal Incident Forecasting & Empirical Model Backtesting")
st.markdown("Holt's Linear Exponential Smoothing with historical out-of-sample error validation.")

df = load_analytical_data()
horizon = st.slider("Forecast Horizon (Years)", 1, 6, 3)

res = generate_incident_forecast(df, forecast_horizon=horizon)

if "error" in res:
    st.error(res["error"])
else:
    eval_m = res["evaluation_metrics"]
    
    # Backtest Metrics Bar
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Backtested MAE", f"{eval_m['mae']} events")
    m2.metric("Backtested RMSE", f"{eval_m['rmse']}")
    m3.metric("Backtested MAPE", f"{eval_m['mape']}%")
    m4.metric("Naive Baseline MAE", f"{eval_m['naive_mae']} events")

    # Time series chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["historical_years"], y=res["historical_counts"], mode="lines+markers", name="Historical Observed", line=dict(color="#58A6FF", width=2)))
    fig.add_trace(go.Scatter(x=res["future_years"], y=res["forecast_counts"], mode="lines+markers", name="Holt's Forecast", line=dict(color="#39D353", width=2, dash="dash")))
    fig.add_trace(go.Scatter(
        x=res["future_years"] + res["future_years"][::-1],
        y=res["upper_bound"] + res["lower_bound"][::-1],
        fill="toself", fillcolor="rgba(57, 211, 83, 0.15)", line=dict(color="rgba(255,255,255,0)"), name="95% Confidence Bounds"
    ))

    fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"**Model Evaluation Note:** Backtesting was conducted across the final {eval_m['evaluated_periods']} historical holdout periods. Holt's Linear achieved a {eval_m['mae']} MAE vs {eval_m['naive_mae']} for the naive persistence baseline.")