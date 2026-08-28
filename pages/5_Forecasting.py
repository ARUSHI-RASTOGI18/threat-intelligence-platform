"""
Module 5: Longitudinal Forecasting, Multi-Model Backtesting & Uncertainty Bounds
Location: ./pages/5_Forecasting.py
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.data_loader import load_analytical_data
from src.forecasting import (
    generate_incident_forecast,
    run_walk_forward_validation
)

st.set_page_config(page_title="Forecasting | GTI-ARP", page_icon="📈", layout="wide")

# Dark Command Center Styling
st.markdown("""
<style>
    .forecast-header {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #58A6FF;
        margin-bottom: 2px;
    }
    .hud-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.95) 0%, rgba(13, 17, 23, 0.98) 100%);
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    }
    .hud-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8B949E;
    }
    .hud-value {
        font-size: 22px;
        font-weight: 800;
        color: #F0F6FC;
        margin-top: 2px;
    }
    .hud-sub {
        font-size: 11px;
        font-weight: 600;
        margin-top: 2px;
    }
    .model-badge {
        background-color: #0E4429;
        border: 1px solid #39D353;
        color: #39D353;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

df = load_analytical_data()

st.markdown("<div class='forecast-header'>📈 Longitudinal Incident Forecasting & Model Backtesting</div>", unsafe_allow_html=True)
st.caption("Out-of-sample holdout benchmarking, expanding-window walk-forward validation, empirical error diagnostics, and aligned horizon projections.")

# Interactive Controls Bar
c1, c2, c3 = st.columns([1.5, 1.5, 1.5])
with c1:
    horizon = st.slider("Forecast Horizon (Years Ahead)", min_value=1, max_value=8, value=4)
with c2:
    model_override = st.selectbox(
        "Active Projection Engine",
        ["Automatic (Best Backtested Model)", "Naive Persistence Baseline", "Moving Average (3-Period MA)", "Simple Exponential Smoothing (SES)", "Holt's Linear Exponential Smoothing"]
    )
    forced_name = None if "Automatic" in model_override else model_override
with c3:
    validation_mode = st.radio("Backtesting Framework", ["Holdout Backtest", "Walk-Forward (Rolling)"], horizontal=True)

# Generate Unified Forecast
res = generate_incident_forecast(df, forecast_horizon=horizon, forced_model=forced_name)

if "error" in res:
    st.error(f"⚠️ {res['error']}")
    st.stop()

backtest = res["backtest_results"]
df_leaderboard = backtest["leaderboard"]

# 1. Forecast Direction & Advantage Telemetry
h1, h2, h3, h4 = st.columns(4)
with h1:
    st.markdown(f"""
    <div class="hud-card">
        <div class="hud-title">Active Projection Engine</div>
        <div class="hud-value" style="font-size:16px;color:#58A6FF;padding-top:4px;">{res['active_model_name']}</div>
        <div class="hud-sub" style="color:{'#39D353' if res['is_optimal_model'] else '#FFA657'};">
            {'● Verified Optimal Backtested Model' if res['is_optimal_model'] else '● Manual Benchmark Override'}
        </div>
    </div>
    """, unsafe_allow_html=True)

with h2:
    st.markdown(f"""
    <div class="hud-card">
        <div class="hud-title">Model Advantage vs Runner-Up</div>
        <div class="hud-value" style="color:#39D353;">+{backtest['mae_advantage_pct']}%</div>
        <div class="hud-sub" style="color:#8B949E;">MAE: {backtest['best_mae']} vs {backtest['runner_up_mae']} events</div>
    </div>
    """, unsafe_allow_html=True)

with h3:
    st.markdown(f"""
    <div class="hud-card">
        <div class="hud-title">Trajectory Direction</div>
        <div class="hud-value" style="font-size:17px;color:{res['direction_color']};padding-top:3px;">{res['forecast_direction']}</div>
        <div class="hud-sub" style="color:{res['direction_color']};">{res['percentage_change']:+,.1f}% vs Last Year</div>
    </div>
    """, unsafe_allow_html=True)

with h4:
    st.markdown(f"""
    <div class="hud-card">
        <div class="hud-title">Residual Uncertainty (±1.96σ)</div>
        <div class="hud-value">±{round(res['residual_std'] * 1.96, 0):,.0f}</div>
        <div class="hud-sub" style="color:#8B949E;">Residual σ = {res['residual_std']} events</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 2. Main Visual Canvas: Unified Trend & Future Projection
st.subheader(f"1. Longitudinal Historical Trajectory & {res['active_model_name']} Horizon")

fig_main = go.Figure()

# Historical actuals
fig_main.add_trace(go.Scatter(
    x=res["historical_years"],
    y=res["historical_counts"],
    mode="lines+markers",
    name="Historical Observed Volume",
    line=dict(color="#58A6FF", width=2.5),
    marker=dict(size=5)
))

# Forecast Extrapolation
fig_main.add_trace(go.Scatter(
    x=res["future_years"],
    y=res["forecast_counts"],
    mode="lines+markers",
    name=f"{res['active_model_name']} Projection",
    line=dict(color="#39D353", width=2.5, dash="dash"),
    marker=dict(size=7, symbol="diamond")
))

# 95% Analytical Uncertainty Bounds
fig_main.add_trace(go.Scatter(
    x=res["future_years"] + res["future_years"][::-1],
    y=res["upper_bound"] + res["lower_bound"][::-1],
    fill="toself",
    fillcolor="rgba(57, 211, 83, 0.12)",
    line=dict(color="rgba(255,255,255,0)"),
    hoverinfo="skip",
    name="95% Analytical Uncertainty Bounds"
))

fig_main.update_layout(
    template="plotly_dark",
    height=420,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title="Calendar Year",
    yaxis_title="Annual Incident Volume",
    legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5)
)
st.plotly_chart(fig_main, use_container_width=True)

# Forecast Schedule Table
t_col1, t_col2 = st.columns([1.6, 1.4])
with t_col1:
    st.markdown("#### Tabular Horizon Projection Schedule")
    df_sched = pd.DataFrame({
        "Projection Year": [str(y) for y in res["future_years"]],
        "Forecast Point Estimate": [f"{int(c):,} events" for c in res["forecast_counts"]],
        "Lower Bound (-1.96σ)": [f"{int(l):,} events" for l in res["lower_bound"]],
        "Upper Bound (+1.96σ)": [f"{int(u):,} events" for u in res["upper_bound"]],
        "Analytical Bandwidth (±)": [f"±{int((u - l) / 2):,} events" for l, u in zip(res["lower_bound"], res["upper_bound"])]
    })
    st.dataframe(df_sched, use_container_width=True, hide_index=True)

with t_col2:
    st.markdown("#### Forecast Directional Summary")
    st.markdown(f"""
    - **Base Historical Year:** `{res['last_historical_year']}` (Volume: `{int(res['last_observed_value']):,}` incidents)
    - **Terminal Forecast Year:** `{res['future_years'][-1]}` (Volume: `{int(res['final_forecast_value']):,}` incidents)
    - **Net Expected Volume Shift:** `{res['absolute_change']:+,.0f}` incidents (`{res['percentage_change']:+,.1f}%`)
    - **Uncertainty Definition:** Analytical bounds represent $\\pm 1.96\\sigma$ of historical 1-step residual errors.
    """)

st.markdown("<br>", unsafe_allow_html=True)

# 3. Model Benchmark Leaderboard & Backtest Trajectory
st.subheader("2. Out-of-Sample Backtest Suite & Validation Leaderboard")

tab_backtest, tab_comparison, tab_diagnostics = st.tabs([
    "🏆 Model Leaderboard & Backtest Chart",
    "📊 Comparative Metric Breakdown",
    "🔬 Residual Error Diagnostics"
])

with tab_backtest:
    if validation_mode == "Holdout Backtest":
        st.markdown(f"**Validation Framework:** Static Holdout ($k = {backtest['holdout_periods']}$ Final Historical Periods: Out-of-Sample Test)")
        st.dataframe(
            df_leaderboard,
            column_config={
                "Rank": "Rank",
                "Model": "Forecasting Model",
                "MAE (Events)": st.column_config.NumberColumn("MAE (Events)", format="%.1f"),
                "RMSE": st.column_config.NumberColumn("RMSE", format="%.1f"),
                "MAPE (%)": st.column_config.NumberColumn("MAPE (%)", format="%.2f%%"),
                "sMAPE (%)": st.column_config.NumberColumn("sMAPE (%)", format="%.2f%%")
            },
            hide_index=True,
            use_container_width=True
        )

        # Backtest Actual vs Predicted Trajectory Plot
        st.markdown("#### Actual vs Model Predictions (Holdout Window)")
        holdout_years = res["historical_years"][-backtest["holdout_periods"]:]
        
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=holdout_years,
            y=backtest["actual_holdout"],
            mode="lines+markers",
            name="Actual Observed",
            line=dict(color="#F0F6FC", width=3)
        ))
        
        colors = ["#39D353", "#FFA657", "#D29922", "#FF7B72"]
        for idx, (m_name, preds) in enumerate(backtest["backtest_predictions"].items()):
            fig_bt.add_trace(go.Scatter(
                x=holdout_years,
                y=preds,
                mode="lines+markers",
                name=m_name,
                line=dict(color=colors[idx % len(colors)], width=2, dash="dot" if m_name != backtest["best_model_name"] else "solid")
            ))

        fig_bt.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Holdout Evaluation Year",
            yaxis_title="Incident Volume",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_bt, use_container_width=True)

    else:
        st.markdown("**Validation Framework:** Expanding-Window Walk-Forward Validation ($t_0..t_k \\rightarrow t_{k+1}$ across historical epochs)")
        counts_arr = np.array(res["historical_counts"])
        df_wf = run_walk_forward_validation(counts_arr, min_train=12)
        st.dataframe(df_wf, use_container_width=True, hide_index=True)

with tab_comparison:
    st.markdown("#### Comparative Error Metrics Across Competing Engines")
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        fig_mae = px.bar(
            df_leaderboard,
            x="MAE (Events)",
            y="Model",
            orientation="h",
            template="plotly_dark",
            color="MAE (Events)",
            color_continuous_scale="Viridis",
            title="Mean Absolute Error (Lower is Better)"
        )
        fig_mae.update_layout(yaxis=dict(autorange="reversed"), height=260, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_mae, use_container_width=True)

    with c_m2:
        fig_smape = px.bar(
            df_leaderboard,
            x="sMAPE (%)",
            y="Model",
            orientation="h",
            template="plotly_dark",
            color="sMAPE (%)",
            color_continuous_scale="Purples",
            title="Symmetric MAPE (%) (Lower is Better)"
        )
        fig_smape.update_layout(yaxis=dict(autorange="reversed"), height=260, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_smape, use_container_width=True)

with tab_diagnostics:
    st.markdown("#### Model In-Sample Residual Error Diagnostics")
    d1, d2, d3 = st.columns(3)
    d1.metric("Mean Residual Error", f"{res['residual_mean']} events")
    d2.metric("Residual Standard Deviation (σ)", f"{res['residual_std']} events")
    d3.metric("Error Sample Depth", f"{len(res['residuals'])} epochs")

    fig_hist = px.histogram(
        x=res["residuals"],
        nbins=15,
        template="plotly_dark",
        title="Historical Residual Error Distribution (Actual - Fitted)",
        color_discrete_sequence=["#58A6FF"]
    )
    fig_hist.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Residual Error", yaxis_title="Count")
    st.plotly_chart(fig_hist, use_container_width=True)

# 4. Methodological Transparency & Limitations
st.markdown("---")
with st.expander("📖 Methodology, Backtesting Design & Academic Limitations", expanded=False):
    st.markdown("""
    * **Zero Lookahead Contamination:** Models are fitted exclusively on training slices ($t < t_{\\text{holdout}}$). Test periods are strictly isolated to evaluate true out-of-sample generalization.
    * **Mathematical Formulations:**
        * $\\text{MAE} = \\frac{1}{n} \\sum |y_t - \\hat{y}_t|$
        * $\\text{RMSE} = \\sqrt{\\frac{1}{n} \\sum (y_t - \\hat{y}_t)^2}$
        * $\\text{sMAPE} = \\frac{100\\%}{n} \\sum \\frac{2 |y_t - \\hat{y}_t|}{|y_t| + |\\hat{y}_t| + \\epsilon}$
    * **Analytical Bounds vs. Prediction Intervals:** The confidence band represents an empirical $\\pm 1.96\\sigma$ Gaussian approximation derived from historical 1-step residual errors.
    * **Non-Operational Notice:** Time-series projections represent statistical extrapolations of historical aggregate patterns. They do not predict specific tactical operations or real-world events.
    """)