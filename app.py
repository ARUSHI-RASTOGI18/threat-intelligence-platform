"""
Global Threat Intelligence & Analytical Risk Platform (GTI-ARP)
Main Application & Executive Command Center
Location: ./app.py
Execution: streamlit run app.py
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.data_loader import load_analytical_data, get_dataset_metadata, audit_dataset_quality
from src.risk_engine import compute_country_risk_index
from src.anomaly_detection import detect_historical_anomalies
from src.forecasting import generate_incident_forecast
from src.analytics import (
    calculate_period_trends,
    get_threat_glance_summary,
    get_strategic_signals,
    get_research_findings,
    get_pipeline_status
)
from src.ml_engine import load_trained_artifacts

# Page Configuration
st.set_page_config(
    page_title="GTI-ARP | Executive Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark-Themed Executive Styles
st.markdown("""
<style>
    .kpi-card {
        background: linear-gradient(135deg, #161B22 0%, #0D1117 100%);
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 12px 14px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .kpi-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8B949E;
        margin-bottom: 2px;
    }
    .kpi-value {
        font-size: 22px;
        font-weight: 800;
        color: #F0F6FC;
    }
    .kpi-sub {
        font-size: 11px;
        color: #58A6FF;
        font-weight: 600;
        margin-top: 3px;
    }
    .glance-card {
        background-color: #161B22;
        border-left: 3px solid #58A6FF;
        border-radius: 6px;
        padding: 10px 14px;
        border-top: 1px solid #30363D;
        border-right: 1px solid #30363D;
        border-bottom: 1px solid #30363D;
        height: 100%;
    }
    .glance-label {
        font-size: 11px;
        text-transform: uppercase;
        color: #8B949E;
        font-weight: 700;
    }
    .glance-value {
        font-size: 14px;
        font-weight: 700;
        color: #E6EDF3;
        margin-top: 2px;
    }
    .signal-box {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .signal-header {
        font-size: 12px;
        font-weight: 700;
        color: #58A6FF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 3px;
    }
    .signal-content {
        font-size: 13px;
        color: #C9D1D9;
        line-height: 1.4;
    }
    .coverage-pill {
        background-color: #0E4429;
        color: #39D353;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }
    .disclaimer-box {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 11px;
        color: #8B949E;
        line-height: 1.5;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 1. Base Data & Global Telemetry Ingestion
df = load_analytical_data()
meta = get_dataset_metadata(df)
quality = audit_dataset_quality(df)
model, _, _, model_meta = load_trained_artifacts()

# 2. Sidebar Global Horizon Filter (Single Source of Truth)
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=56)
    st.title("GTI-ARP Engine")
    st.caption("AI Threat Intelligence & Risk Platform")
    st.markdown("---")
    
    st.markdown(f"**Data Corpus Span:** <span class='coverage-pill'>{meta['coverage_label']}</span>", unsafe_allow_html=True)
    st.markdown(f"**Data Quality Index:** `{quality['data_quality_score']}/100`")
    
    m_acc = model_meta.get("best_model_metrics", {}).get("accuracy", "N/A") if model_meta else "N/A"
    top_model_name = model_meta.get('best_model_name', 'Random Forest').split('(')[0] if model_meta else 'Random Forest'
    st.markdown(f"**Primary Model:** `{top_model_name}`")
    st.markdown(f"**Model Accuracy:** `{m_acc}%`")
    st.markdown("---")
    
    min_dataset_yr = meta['min_year']
    max_dataset_yr = meta['max_year']
    
    if min_dataset_yr < max_dataset_yr:
        selected_years = st.slider(
            "Global Horizon Window",
            min_value=min_dataset_yr,
            max_value=max_dataset_yr,
            value=(min_dataset_yr, max_dataset_yr),
            key="global_horizon_slider"
        )
    else:
        selected_years = (min_dataset_yr, max_dataset_yr)

# 3. Derive Filtered Dataset State
df_filtered = df[(df["year"] >= selected_years[0]) & (df["year"] <= selected_years[1])].copy()

# 4. Compute ALL Dependent Analytics from the FILTERED slice
risk_df_filtered = compute_country_risk_index(df_filtered)
trends_filtered = calculate_period_trends(df_filtered)
anomalies_filtered = detect_historical_anomalies(df_filtered, window=min(5, max(2, len(df_filtered["year"].unique()) - 1)))
glance_filtered = get_threat_glance_summary(df_filtered, risk_df_filtered)
signals_filtered = get_strategic_signals(df_filtered, risk_df_filtered, trends_filtered, model_meta)

# Header Title
st.title("Executive Threat Intelligence Command Center")
st.markdown(
    f"Multi-model tactical classification, statistical anomaly surveillance & longitudinal risk forecasting. "
    f"**Active Horizon: {selected_years[0]} – {selected_years[1]}**"
)

if df_filtered.empty:
    st.error("No historical records match the selected horizon window. Please expand the slider range.")
    st.stop()

# 5. Executive 6-KPI Metric Matrix (100% Calculated from Filtered Slice)
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Incidents</div>
        <div class="kpi-value">{len(df_filtered):,}</div>
        <div class="kpi-sub">{(len(df_filtered)/max(1, len(df)))*100:.1f}% of Total Corpus</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    total_fat = int(df_filtered['fatalities'].sum())
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Fatalities</div>
        <div class="kpi-value" style="color:#F85149;">{total_fat:,}</div>
        <div class="kpi-sub" style="color:#FFA657;">{(total_fat/max(1, len(df_filtered))):.2f} / Incident</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    total_inj = int(df_filtered['injured'].sum())
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Injured</div>
        <div class="kpi-value" style="color:#D29922;">{total_inj:,}</div>
        <div class="kpi-sub" style="color:#D29922;">{(total_inj/max(1, len(df_filtered))):.2f} / Incident</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Active Territories</div>
        <div class="kpi-value" style="color:#58A6FF;">{df_filtered['country'].nunique()}</div>
        <div class="kpi-sub">{df_filtered['region'].nunique()} Major Regions</div>
    </div>
    """, unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Data Quality</div>
        <div class="kpi-value" style="color:#39D353;">{quality['data_quality_score']}%</div>
        <div class="kpi-sub" style="color:#39D353;">{quality['geocoding_coverage_pct']}% Geocoded</div>
    </div>
    """, unsafe_allow_html=True)

with k6:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Top ML Model</div>
        <div class="kpi-value" style="font-size:16px;padding-top:4px;">{top_model_name}</div>
        <div class="kpi-sub">{m_acc}% Holdout Acc</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 6. Threat Intelligence at a Glance (Filtered HUD)
st.subheader("⚡ Threat Intelligence at a Glance")

g1, g2, g3, g4, g5, g6 = st.columns(6)
with g1:
    st.markdown(f"""<div class="glance-card"><div class="glance-label">Peak Threat Territory</div><div class="glance-value">{glance_filtered['highest_risk_country']} ({glance_filtered['highest_risk_score']})</div></div>""", unsafe_allow_html=True)
with g2:
    st.markdown(f"""<div class="glance-card"><div class="glance-label">Dominant Tactic</div><div class="glance-value">{glance_filtered['dominant_tactic']}</div></div>""", unsafe_allow_html=True)
with g3:
    st.markdown(f"""<div class="glance-card"><div class="glance-label">Primary Weapon</div><div class="glance-value">{glance_filtered['primary_weapon']}</div></div>""", unsafe_allow_html=True)
with g4:
    st.markdown(f"""<div class="glance-card"><div class="glance-label">Top Target Sector</div><div class="glance-value">{glance_filtered['top_target']}</div></div>""", unsafe_allow_html=True)
with g5:
    st.markdown(f"""<div class="glance-card"><div class="glance-label">Peak Activity Year</div><div class="glance-value">{glance_filtered['peak_year']} ({glance_filtered['peak_year_count']:,} Events)</div></div>""", unsafe_allow_html=True)
with g6:
    st.markdown(f"""<div class="glance-card"><div class="glance-label">Trajectory Velocity</div><div class="glance-value">{trends_filtered['trend_direction']} ({trends_filtered['incident_delta']:+,.1f}%)</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 7. Global Strategic Signals (Dynamic)
st.subheader("📡 Global Strategic Signals")
sig_cols = st.columns(len(signals_filtered)) if signals_filtered else [st.container()]
for col, sig in zip(sig_cols, signals_filtered):
    with col:
        st.markdown(f"""
        <div class="signal-box">
            <div class="signal-header">● {sig['title']}</div>
            <div class="signal-content">{sig['body']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 8. Dynamic Historical Trajectory & Forecast Outlook
row1_left, row1_right = st.columns([3, 2])

with row1_left:
    st.subheader(f"📈 Historical Incident Trajectory ({selected_years[0]}–{selected_years[1]})")
    yearly_inc = df_filtered.groupby("year").size().reset_index(name="Incidents").sort_values("year")
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=yearly_inc["year"],
        y=yearly_inc["Incidents"],
        mode="lines+markers",
        name="Annual Incidents",
        line=dict(color="#58A6FF", width=2.5),
        marker=dict(size=5)
    ))

    # Dynamic Anomaly Markers
    if not anomalies_filtered.empty:
        anom_pts = anomalies_filtered[anomalies_filtered["is_anomaly"]]
        if not anom_pts.empty:
            fig_trend.add_trace(go.Scatter(
                x=anom_pts["year"],
                y=anom_pts["incident_count"],
                mode="markers",
                name="Surge Anomaly (Z >= 2.0)",
                marker=dict(color="#F85149", size=10, symbol="diamond", line=dict(color="#FFFFFF", width=1)),
                hoverinfo="text",
                hovertext=[f"<b>Year: {int(yr)}</b><br>Volume: {int(cnt):,}<br>Z-Score: {z:.2f}" for yr, cnt, z in zip(anom_pts["year"], anom_pts["incident_count"], anom_pts["z_score"])]
            ))

    # Dynamic Peak Year Annotation
    if not yearly_inc.empty:
        peak_row = yearly_inc.loc[yearly_inc["Incidents"].idxmax()]
        fig_trend.add_annotation(
            x=peak_row["year"],
            y=peak_row["Incidents"],
            text=f"Peak: {int(peak_row['year'])} ({int(peak_row['Incidents']):,} Events)",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#39D353",
            font=dict(color="#39D353", size=11),
            bgcolor="rgba(22, 27, 34, 0.85)",
            bordercolor="#30363D"
        )

    fig_trend.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=30, b=20),
        height=340,
        xaxis_title="Calendar Year",
        yaxis_title="Incident Volume",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with row1_right:
    st.subheader("🔮 Longitudinal Model Forecast Outlook")
    st.caption("Extrapolated dynamically from the active historical series using exponential smoothing.")
    
    # Recalculate forecast using filtered historical series
    forecast_res = generate_incident_forecast(df_filtered, forecast_horizon=min(4, max(1, len(yearly_inc) // 4)))
    
    if "error" not in forecast_res:
        f_hist_yrs = forecast_res["historical_years"][-8:]
        f_hist_cnt = forecast_res["historical_counts"][-8:]
        f_fut_yrs = forecast_res["future_years"]
        f_fut_cnt = forecast_res["forecast_counts"]
        f_up = forecast_res["upper_bound"]
        f_low = forecast_res["lower_bound"]

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=f_hist_yrs, y=f_hist_cnt, mode="lines+markers", name="Observed", line=dict(color="#58A6FF", width=2)))
        fig_fc.add_trace(go.Scatter(x=f_fut_yrs, y=f_fut_cnt, mode="lines+markers", name=f"{forecast_res.get('active_model_name', 'Holt')} Forecast", line=dict(color="#39D353", width=2, dash="dash")))
        fig_fc.add_trace(go.Scatter(
            x=f_fut_yrs + f_fut_yrs[::-1],
            y=f_up + f_low[::-1],
            fill="toself",
            fillcolor="rgba(57, 211, 83, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Analytical Bounds"
        ))
        fig_fc.update_layout(
            template="plotly_dark",
            height=340,
            margin=dict(l=10, r=10, t=30, b=20),
            xaxis_title="Year",
            yaxis_title="Events",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_fc, use_container_width=True)
    else:
        st.info("Insufficient longitudinal points in active window for statistical forecasting (Requires >= 5 historical years).")

st.markdown("<br>", unsafe_allow_html=True)

# 9. Top Threat Concentrations (Derived Directly from Filtered Slice)
c_left, c_mid, c_right = st.columns([1.4, 1.3, 1.3])

with c_left:
    st.subheader(f"🎯 Top Threat Concentrations ({selected_years[0]}–{selected_years[1]})")
    if not risk_df_filtered.empty:
        top_5_risk = risk_df_filtered.head(5)[["country", "total_incidents", "total_fatalities", "composite_risk_score", "risk_level"]]
        st.dataframe(
            top_5_risk,
            column_config={
                "country": "Territory",
                "total_incidents": st.column_config.NumberColumn("Incidents", format="%d"),
                "total_fatalities": st.column_config.NumberColumn("Deaths", format="%d"),
                "composite_risk_score": st.column_config.NumberColumn("Threat Score", format="%.1f"),
                "risk_level": "Tier"
            },
            hide_index=True,
            use_container_width=True
        )
    st.caption("Calculated dynamically across active filtered records.")

with c_mid:
    st.subheader("🤖 Model Intelligence Summary")
    if model_meta and "best_model_metrics" in model_meta:
        b_mets = model_meta["best_model_metrics"]
        st.markdown(f"""
        - **Selected Algorithm:** `{model_meta.get('best_model_name')}`
        - **Validation Protocol:** `{model_meta.get('validation_strategy')}`
        - **Test Accuracy:** `{b_mets.get('accuracy')}%`
        - **Macro-F1 Score:** `{b_mets.get('macro_f1')}%`
        - **Weighted-F1 Score:** `{b_mets.get('weighted_f1')}%`
        - **Training Corpus:** `{model_meta.get('train_samples'):,}` instances
        """)
    else:
        st.warning("Model metadata uninitialized. Execute `python train_pipeline.py`.")

with c_right:
    st.subheader("🛡️ Data Trust & Integrity")
    st.markdown(f"""
    - **Active Filtered Records:** `{len(df_filtered):,}` / `{meta['total_records']:,}` total
    - **Feature Completeness:** `{quality['completeness_pct']}%`
    - **Geocoding Validity:** `{quality['geocoding_coverage_pct']}%`
    - **Temporal Span:** `{meta['coverage_label']}` (Current: `{selected_years[0]}–{selected_years[1]}`)
    - **Memory Cache Footprint:** `{quality['memory_mb']} MB (Parquet)`
    """)

st.markdown("<br>", unsafe_allow_html=True)

# 10. Statistical Anomaly Watch & System Operational Status
anom_col, pipe_col = st.columns([1.2, 1.8])

with anom_col:
    st.subheader("⚡ Statistical Anomaly Watch")
    if not anomalies_filtered.empty:
        flagged_anom = anomalies_filtered[anomalies_filtered["is_anomaly"]]
        anom_total = len(flagged_anom)
        peak_anom_row = flagged_anom.loc[flagged_anom["z_score"].idxmax()] if not flagged_anom.empty else None
        peak_anom_yr = int(peak_anom_row["year"]) if peak_anom_row is not None else "None"
        peak_z = round(float(peak_anom_row["z_score"]), 2) if peak_anom_row is not None else 0.0

        st.markdown(f"""
        - **Detected Surges in Horizon:** `{anom_total}` periods
        - **Peak Statistical Surge:** Year **{peak_anom_yr}** ($Z = {peak_z}$)
        - **Baseline Window:** {min(5, max(2, len(df_filtered['year'].unique()) - 1))}-Year Rolling Mean
        - **Surveillance Threshold:** $|Z| \\ge 2.0$ Standard Deviations
        """)
    else:
        st.info("Insufficient longitudinal points in active window for anomaly detection.")

with pipe_col:
    st.subheader("⚙️ Analytical Pipeline Operational Status")
    status_list = get_pipeline_status(model_meta, df_filtered)
    
    p_cols = st.columns(3)
    for idx, item in enumerate(status_list):
        with p_cols[idx % 3]:
            st.markdown(f"""
            <div style="background-color:#161B22;border:1px solid #30363D;border-radius:6px;padding:8px;margin-bottom:6px;text-align:center;">
                <div style="font-size:10px;color:#8B949E;font-weight:700;">{item['module']}</div>
                <div style="font-size:12px;font-weight:800;color:{item['color']};">● {item['status']}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 11. Empirical Research Findings (Filtered Timeframe)
st.subheader("🔬 Key Empirical Research Findings")
findings = get_research_findings(df_filtered, risk_df_filtered, model_meta, quality)
for finding in findings:
    st.markdown(f"- {finding}")

st.markdown("<br>", unsafe_allow_html=True)

# 12. Academic Disclaimer Footer
st.markdown(f"""
<div class="disclaimer-box">
    <b>Academic Research Prototype & Research Ethics Disclosure:</b><br>
    This platform operates exclusively for academic, statistical, and retrospective research on historical event data ({selected_years[0]}–{selected_years[1]}). 
    Machine learning classifications, deterministic risk indices, and longitudinal forecasts represent model-based statistical estimates. 
    They do not constitute real-time operational intelligence, tactical security guidance, or future event targeting.
</div>
""", unsafe_allow_html=True)