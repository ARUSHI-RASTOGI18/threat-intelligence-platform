"""
Global Threat Intelligence & Analytical Risk Platform (GTI-ARP)
System Architecture & Data Flow Pipeline
Location: ./pages/10_System_Architecture.py
"""

import streamlit as st

st.set_page_config(
    page_title="GTI-ARP | System Architecture",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ System Architecture & Data Flow Pipeline")
st.caption("End-to-end analytical pipeline topology, modular subsystem dependencies, and AI taxonomy.")

# 1. High-Resolution Architecture Flowchart (Native Streamlit Graphviz Engine)
arch_dot = """
digraph GTI_ARP_Architecture {
    graph [
        rankdir=TB,
        bgcolor="#0E1117",
        fontname="Helvetica,Arial,sans-serif",
        pad="0.5",
        nodesep="0.6",
        ranksep="0.7",
        compound=true
    ];

    node [
        fontname="Helvetica,Arial,sans-serif",
        fontsize=11,
        shape=box,
        style="filled,rounded",
        color="#30363D",
        fontcolor="#FFFFFF",
        penwidth=1.5,
        margin="0.25,0.18"
    ];

    edge [
        fontname="Helvetica,Arial,sans-serif",
        fontsize=9,
        fontcolor="#8B949E",
        color="#58A6FF",
        arrowsize=0.85,
        penwidth=1.8
    ];

    // Subgraph 1: Data Ingestion & Quality Layer
    subgraph cluster_ingestion {
        label="LAYER 1: DATA INGESTION & QUALITY ASSURANCE";
        fontcolor="#58A6FF";
        color="#21262D";
        style="filled,rounded";
        fillcolor="#161B22";
        fontsize=12;

        raw_gtd [label="📂 GTD Corpus (1970–2017)\\n[181,691 Validated Events]", fillcolor="#1F242C", color="#58A6FF"];
        quality_audit [label="🛡️ Quality Audit & Schema Harmonizer\\n[Coordinate Imputation & Validation]", fillcolor="#21262D", color="#30363D"];
        parquet_cache [label="⚡ High-Performance Parquet Cache\\n[Optimized Arrow IPC]", fillcolor="#21262D", color="#30363D"];

        raw_gtd -> quality_audit [label=" Raw Records"];
        quality_audit -> parquet_cache [label=" Normalized Schema"];
    }

    // Subgraph 2: Processing & Computation Engines
    subgraph cluster_engines {
        label="LAYER 2: DUAL ANALYTICAL & PREDICTIVE PIPELINE";
        fontcolor="#39D353";
        color="#21262D";
        style="filled,rounded";
        fillcolor="#161B22";
        fontsize=12;

        // Deterministic Branch
        risk_engine [label="📐 Composite Risk Engine\\n[Threat Index (0-100) & Log Normalization]", fillcolor="#1C2D27", color="#39D353"];
        anomaly_engine [label="📊 Anomaly Surveillance\\n[5-Year Rolling Window Z-Score (Z >= 2.0)]", fillcolor="#1C2D27", color="#39D353"];
        forecasting_engine [label="📈 Longitudinal Forecaster\\n[Holt's Linear Exponential Smoothing]", fillcolor="#1C2D27", color="#39D353"];

        // Supervised ML Branch
        ml_pipeline [label="🤖 Supervised ML Engine\\n[Temporal Holdout Validation (70/15/15)]", fillcolor="#2E1F3B", color="#BC8CFF"];
        model_zoo [label="🧪 Multi-Model Benchmark Suite\\n[RF, HistGradientBoosting, Logistic Reg]", fillcolor="#2E1F3B", color="#BC8CFF"];
        interpretability [label="🔍 Model Explainability\\n[Permutation Feature Importance]", fillcolor="#2E1F3B", color="#BC8CFF"];

        ml_pipeline -> model_zoo -> interpretability;
    }

    // Subgraph 3: Interactive Simulation
    subgraph cluster_simulation {
        label="LAYER 3: SCENARIO & COUNTERFACTUAL SIMULATION";
        fontcolor="#FFA657";
        color="#21262D";
        style="filled,rounded";
        fillcolor="#161B22";
        fontsize=12;

        what_if [label="🎯 Interactive What-If Simulator\\n[Casualty & Tactical Feature Shifts]", fillcolor="#2D2318", color="#FFA657"];
        counterfactual [label="⚡ Real-Time Probability Re-estimation\\n[Pre-event Tactical Response Curves]", fillcolor="#2D2318", color="#FFA657"];

        what_if -> counterfactual;
    }

    // Subgraph 4: Presentation & Synthesis
    subgraph cluster_presentation {
        label="LAYER 4: EXECUTIVE DECISION SUPPORT & PRESENTATION";
        fontcolor="#58A6FF";
        color="#21262D";
        style="filled,rounded";
        fillcolor="#161B22";
        fontsize=12;

        dashboard_hud [label="🖥️ Executive Command Center (HUD)\\n[Plotly Dark Visuals & Geospatial GIS]", fillcolor="#1F242C", color="#58A6FF"];
        ai_briefing [label="📑 Deterministic AI Synthesis Brief\\n[Structured 13-Section Academic Report]", fillcolor="#1F242C", color="#58A6FF"];
    }

    // Cross-Layer Pipeline Connections
    parquet_cache -> risk_engine [lhead=cluster_engines, color="#58A6FF"];
    parquet_cache -> ml_pipeline [color="#BC8CFF"];

    risk_engine -> what_if [lhead=cluster_simulation, color="#FFA657"];
    model_zoo -> what_if [color="#FFA657"];

    risk_engine -> dashboard_hud [lhead=cluster_presentation, color="#39D353"];
    anomaly_engine -> dashboard_hud [color="#39D353"];
    forecasting_engine -> dashboard_hud [color="#39D353"];
    interpretability -> dashboard_hud [color="#BC8CFF"];
    counterfactual -> dashboard_hud [color="#FFA657"];
    risk_engine -> ai_briefing [color="#58A6FF"];
}
"""

st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# 2. Academic Taxonomy Section
st.subheader("🎓 Academic Taxonomy: AI vs Deterministic Analytics")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div style="background:#161B22; border:1px solid #30363D; border-top:3px solid #BC8CFF; border-radius:8px; padding:16px;">
        <div style="font-size:14px; font-weight:700; color:#BC8CFF; text-transform:uppercase; margin-bottom:8px;">🤖 Supervised Machine Learning</div>
        <ul style="font-size:12px; color:#C9D1D9; padding-left:16px; line-height:1.6; margin-bottom:0;">
            <li><b>Task:</b> Multiclass tactical vector classification from pre-event indicators.</li>
            <li><b>Architectures:</b> RandomForest, HistGradientBoosting, Logistic Regression.</li>
            <li><b>Validation Protocol:</b> Three-way Temporal Holdout Split (Train/Val/Test) preventing retrospective data leakage.</li>
            <li><b>Explainability:</b> Permutation Feature Importance ranking.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div style="background:#161B22; border:1px solid #30363D; border-top:3px solid #39D353; border-radius:8px; padding:16px;">
        <div style="font-size:14px; font-weight:700; color:#39D353; text-transform:uppercase; margin-bottom:8px;">📐 Deterministic Analytics</div>
        <ul style="font-size:12px; color:#C9D1D9; padding-left:16px; line-height:1.6; margin-bottom:0;">
            <li><b>Threat Risk Index:</b> 0–100 composite formula factoring incident volume and log-normal casualty severity.</li>
            <li><b>Scenario Simulator:</b> Exact mathematical recalculation based on user parameter adjustments.</li>
            <li><b>Spatial Aggregation:</b> Density clustering and hotspot coordinate indexing.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div style="background:#161B22; border:1px solid #30363D; border-top:3px solid #FFA657; border-radius:8px; padding:16px;">
        <div style="font-size:14px; font-weight:700; color:#FFA657; text-transform:uppercase; margin-bottom:8px;">📈 Statistical & Time-Series</div>
        <ul style="font-size:12px; color:#C9D1D9; padding-left:16px; line-height:1.6; margin-bottom:0;">
            <li><b>Anomaly Detection:</b> 5-year rolling window Z-scores ($Z \\ge 2.0$) for surge identification.</li>
            <li><b>Trend Velocity:</b> Multi-period percentage delta calculations.</li>
            <li><b>Forecasting:</b> Holt's Linear Double Exponential Smoothing with trend damping.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)