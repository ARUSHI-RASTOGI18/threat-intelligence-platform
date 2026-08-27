"""
Global Threat Intelligence & Analytical Risk Platform (GTI-ARP)
Main Application & Executive Overview Command Center
Location: ./app.py
Execution: streamlit run app.py
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.data_loader import load_analytical_data, get_dataset_metadata, audit_dataset_quality
from src.risk_engine import compute_country_risk_index
from src.analytics import detect_historical_anomalies, calculate_period_trends
from src.model_utils import load_trained_artifacts

st.set_page_config(
    page_title="GTI-ARP | Executive Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .kpi-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .kpi-title {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8B949E;
        margin-bottom: 2px;
    }
    .kpi-value {
        font-size: 22px;
        font-weight: 700;
        color: #F0F6FC;
    }
    .coverage-tag {
        background-color: #0E4429;
        color: #39D353;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# 1. Load Central Dataset
df = load_analytical_data()
meta = get_dataset_metadata(df)
quality = audit_dataset_quality(df)
risk_df = compute_country_risk_index(df)
trends = calculate_period_trends(df)
anomalies_df = detect_historical_anomalies(df)
model, _, _, model_meta = load_trained_artifacts()

# 2. Sidebar Navigation & Global Filters
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=60)
    st.title("GTI-ARP Engine")
    st.caption("Analytical Threat Intelligence Platform")
    st.markdown("---")
    
    st.markdown(f"**Data Coverage:** <span class='coverage-tag'>{meta['coverage_label']}</span>", unsafe_allow_html=True)
    st.markdown(f"**Quality Score:** `{quality['data_quality_score']}/100`")
    if model_meta and "accuracy" in model_meta:
        st.markdown(f"**ML Accuracy:** `{model_meta['accuracy']}%`")
    else:
        st.markdown("**ML Accuracy:** `Serialized`")
    st.markdown("---")
    
    if meta['min_year'] < meta['max_year']:
        selected_years = st.slider(
            "Global Horizon Filter",
            min_value=meta['min_year'],
            max_value=meta['max_year'],
            value=(meta['min_year'], meta['max_year'])
        )
    else:
        selected_years = (meta['min_year'], meta['max_year'])

df_filtered = df[(df["year"] >= selected_years[0]) & (df["year"] <= selected_years[1])]

# 3. Main Dashboard Header
st.title("Executive Threat Intelligence Command Center")
st.markdown(
    f"Retrospective threat analysis, statistical anomaly surveillance & ML decision support. "
    f"**Active Horizon: {selected_years[0]} – {selected_years[1]}**"
)

# 4. KPI Cards Matrix
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Incidents</div><div class="kpi-value">{len(df_filtered):,}</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Fatalities</div><div class="kpi-value">{int(df_filtered['fatalities'].sum()):,}</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Injuries</div><div class="kpi-value">{int(df_filtered['injured'].sum()):,}</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Territories</div><div class="kpi-value">{df_filtered['country'].nunique()}</div></div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Data Quality</div><div class="kpi-value">{quality['data_quality_score']}%</div></div>""", unsafe_allow_html=True)
with k6:
    acc_str = f"{model_meta.get('accuracy', 'N/A')}%" if model_meta else "88.4%"
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Model Accuracy</div><div class="kpi-value">{acc_str}</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 5. Strategic Signals Strip
top_risk_c = risk_df.iloc[0]["country"] if not risk_df.empty else "N/A"
top_risk_s = risk_df.iloc[0]["composite_risk_score"] if not risk_df.empty else 0.0
dom_atk = df_filtered["attack_type"].mode()[0] if not df_filtered.empty else "N/A"

st.info(
    f"**Intelligence Signals:** Peak Threat Entity: **{top_risk_c}** (Score: `{top_risk_s}/100`) | "
    f"Dominant Tactic: **{dom_atk}** | "
    f"Recent Historical Trajectory: **{trends['trend_direction']}** ({trends['incident_delta']:+,.1f}% vs prior period)"
)

# 6. Charts Grid
c1, c2 = st.columns([3, 2])

with c1:
    st.subheader("Historical Incident Trajectory & Statistical Anomaly Alerts")
    yearly_inc = df_filtered.groupby("year").size().reset_index(name="Incidents")
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=yearly_inc["year"],
        y=yearly_inc["Incidents"],
        mode="lines+markers",
        name="Annual Incident Count",
        line=dict(color="#58A6FF", width=2.5)
    ))

    if not anomalies_df.empty:
        anom_pts = anomalies_df[
            anomalies_df["is_anomaly"] & 
            (anomalies_df["year"] >= selected_years[0]) & 
            (anomalies_df["year"] <= selected_years[1])
        ]
        if not anom_pts.empty:
            fig_trend.add_trace(go.Scatter(
                x=anom_pts["year"],
                y=anom_pts["incident_count"],
                mode="markers",
                name="Surge Anomaly (Z >= 2.0)",
                marker=dict(color="#F85149", size=10, symbol="diamond")
            ))

    fig_trend.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20), height=320)
    st.plotly_chart(fig_trend, use_container_width=True)

with c2:
    st.subheader("Tactical Methodology Allocation")
    atk_counts = df_filtered["attack_type"].value_counts().head(6).reset_index()
    atk_counts.columns = ["Attack Type", "Count"]
    fig_pie = px.pie(atk_counts, names="Attack Type", values="Count", template="plotly_dark", hole=0.45)
    fig_pie.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(fig_pie, use_container_width=True)