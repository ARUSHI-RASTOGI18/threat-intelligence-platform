"""
Module 6: Statistical Anomaly & Surge Surveillance
Location: ./pages/6_Anomaly_Detection.py
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from src.data_loader import load_analytical_data
from src.anomaly_detection import detect_historical_anomalies

st.set_page_config(page_title="Anomaly Detection | GTI-ARP", page_icon="⚡", layout="wide")

# Dark command-center styling matching the platform theme
st.markdown("""
<style>
    .anomaly-header {
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
    .logic-box {
        background-color: #161B22;
        border-left: 3px solid #58A6FF;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 12px;
        color: #C9D1D9;
        line-height: 1.4;
        border-top: 1px solid #30363D;
        border-right: 1px solid #30363D;
        border-bottom: 1px solid #30363D;
    }
    .summary-box {
        background: linear-gradient(90deg, rgba(14, 68, 41, 0.35) 0%, rgba(22, 27, 34, 0.8) 100%);
        border-left: 4px solid #39D353;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 13px;
        color: #E6EDF3;
        margin-top: 14px;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)

df = load_analytical_data()

st.markdown("<div class='anomaly-header'>⚡ Statistical Anomaly & Surge Surveillance</div>", unsafe_allow_html=True)
st.caption("Longitudinal outlier detection and multi-period surge identification using rolling baseline Z-score statistics.")

# 1. Parameter Controls
c1, c2 = st.columns(2)
with c1:
    window_val = st.slider("Rolling Baseline Window (Years)", 3, 10, 8)
with c2:
    z_val = st.slider("Z-Score Sensitivity Threshold", 1.5, 3.5, 2.0, step=0.1)

anom_df = detect_historical_anomalies(df, window=window_val, z_threshold=z_val)

if not anom_df.empty:
    flagged = anom_df[anom_df["is_anomaly"]].copy()
    anom_count = len(flagged)

    # 2. KPI & Dynamic Detection Logic Panel
    kpi_col, logic_col = st.columns([1.2, 2.8])
    with kpi_col:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Historical Anomalies</div>
            <div class="hud-value" style="color: {'#F85149' if anom_count > 0 else '#39D353'};">{anom_count} anomalies detected</div>
            <div class="hud-sub" style="color: #8B949E;">Across {len(anom_df)} Historical Years</div>
        </div>
        """, unsafe_allow_html=True)

    with logic_col:
        st.markdown(f"""
        <div class="logic-box">
            <b>Detection Logic:</b> Historical years with <code>Z-score ≥ {z_val:.2f}</code> are flagged as statistical anomalies against a 
            <b>{window_val}-year rolling historical baseline</b>. Outliers indicate statistically significant surge periods rather than gradual baseline trend growth.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Anomaly Trajectory Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=anom_df["year"],
        y=anom_df["incident_count"],
        mode="lines+markers",
        name="Observed Incident Volume",
        line=dict(color="#58A6FF", width=2.5),
        marker=dict(size=5)
    ))
    fig.add_trace(go.Scatter(
        x=anom_df["year"],
        y=anom_df["rolling_mean"],
        mode="lines",
        name=f"{window_val}-Year Rolling Baseline",
        line=dict(color="#8B949E", dash="dot", width=1.8)
    ))

    if not flagged.empty:
        fig.add_trace(go.Scatter(
            x=flagged["year"],
            y=flagged["incident_count"],
            mode="markers",
            name=f"Statistical Surge (Z ≥ {z_val:.1f})",
            marker=dict(color="#F85149", size=10, symbol="diamond", line=dict(color="#FFFFFF", width=1)),
            hoverinfo="text",
            hovertext=[
                f"<b>Year: {int(yr)}</b><br>Incidents: {int(cnt):,}<br>Baseline: {int(rm):,}<br>Deviation: {dev:+,.1f}%<br>Z-Score: {z:.2f}<br>Severity: {sev}"
                for yr, cnt, rm, dev, z, sev in zip(
                    flagged["year"], flagged["incident_count"], flagged["rolling_mean"],
                    flagged["pct_deviation"], flagged["z_score"], flagged["anomaly_severity"]
                )
            ]
        ))

    fig.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Calendar Year",
        yaxis_title="Annual Incident Volume",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4. Dynamic Analytical Finding Summary
    if not flagged.empty:
        top_anom_row = flagged.loc[flagged["z_score"].idxmax()]
        peak_anom_yr = int(top_anom_row["year"])
        peak_dev = float(top_anom_row["pct_deviation"])
        peak_cnt = int(top_anom_row["incident_count"])
        peak_base = int(top_anom_row["rolling_mean"])

        st.markdown(f"""
        <div class="summary-box">
            <b>📊 Key Analytical Finding:</b> <b>{anom_count} significant historical surges</b> were detected using the current sensitivity parameters. 
            The strongest statistical deviation occurred in <b>{peak_anom_yr}</b>, logging <b>{peak_cnt:,} events</b> (approximately 
            <b>{peak_dev:+,.1f}%</b> above its {window_val}-year rolling baseline of {peak_base:,} events).
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="summary-box">
            <b>📊 Key Analytical Finding:</b> Zero historical periods exceeded the <code>Z ≥ {z_val:.2f}</code> threshold against the {window_val}-year baseline. All recorded longitudinal volumes remained within expected historical dispersion bounds.
        </div>
        """, unsafe_allow_html=True)

    # 5. Detailed Anomaly Log Table
    st.subheader("Anomaly Log Breakdown")
    if not flagged.empty:
        # Sort chronologically or by peak deviation
        display_df = flagged[[
            "year", "incident_count", "rolling_mean", "z_score", "pct_deviation", "anomaly_severity"
        ]].copy()
        
        display_df["rolling_mean"] = display_df["rolling_mean"].round(1)
        display_df["z_score"] = display_df["z_score"].round(2)

        st.dataframe(
            display_df,
            column_config={
                "year": st.column_config.NumberColumn("Year", format="%d"),
                "incident_count": st.column_config.NumberColumn("Observed Volume", format="%d events"),
                "rolling_mean": st.column_config.NumberColumn(f"{window_val}-Yr Rolling Mean", format="%.1f"),
                "z_score": st.column_config.NumberColumn("Z-Score", format="%.2f"),
                "pct_deviation": st.column_config.NumberColumn("Baseline Deviation", format="%+.1f%%"),
                "anomaly_severity": st.column_config.TextColumn(
                    "Severity Level",
                    help="Categorized based on Z-score deviation bounds."
                )
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No anomaly records to display for the active threshold configuration.")
else:
    st.warning("Insufficient longitudinal records to compute statistical anomalies.")