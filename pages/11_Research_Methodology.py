"""
Module 11: System Methodology & Platform Technical Specifications
Location: ./pages/11_Research_Methodology.py
"""

import streamlit as st
import pandas as pd
from src.data_loader import load_analytical_data, get_dataset_metadata
from src.ml_engine import load_trained_artifacts
from src.risk_engine import (
    WEIGHT_FREQUENCY,
    WEIGHT_FATALITY,
    WEIGHT_INJURY,
    WEIGHT_VELOCITY,
    WEIGHT_DIVERSITY
)

st.set_page_config(page_title="Methodology & Specs | GTI-ARP", page_icon="📑", layout="wide")

# Enterprise Dark-Theme Architecture Styling
st.markdown("""
<style>
    .spec-header {
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
        padding: 14px 18px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    }
    .flow-step {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 6px;
        padding: 10px 12px;
        text-align: center;
        font-size: 12px;
        font-weight: 700;
        color: #E6EDF3;
        border-top: 2px solid #58A6FF;
    }
    .formula-card {
        background-color: #0D1117;
        border: 1px solid #30363D;
        border-left: 4px solid #58A6FF;
        border-radius: 6px;
        padding: 16px 20px;
        margin: 12px 0;
    }
    .tag-badge {
        background-color: rgba(56, 139, 253, 0.15);
        color: #58A6FF;
        border: 1px solid rgba(56, 139, 253, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

df = load_analytical_data()
meta = get_dataset_metadata(df)
model, _, _, model_meta = load_trained_artifacts()

# 1. Page Header
st.markdown("<div class='spec-header'>📑 System Methodology & Technical Specifications</div>", unsafe_allow_html=True)
st.caption("Platform architecture standards, mathematical formulations, feature governance, and longitudinal extrapolation protocols.")

# 2. Platform Specs at a Glance
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("1. Core Platform Technical Specifications")

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f"""
    <div class="hud-card">
        <div style="font-size:11px;font-weight:700;color:#8B949E;text-transform:uppercase;">Data Layer & Storage</div>
        <div style="font-size:15px;font-weight:800;color:#58A6FF;margin-top:2px;">Parquet Analytical Cache</div>
        <div style="font-size:12px;color:#C9D1D9;margin-top:4px;">
            Corpus: <b>{meta['total_records']:,}</b> Events &nbsp;|&nbsp; <b>{meta['coverage_label']}</b><br>
            Sovereign Entities: <b>{meta['unique_countries']}</b> Territories
        </div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    val_strat = model_meta.get("validation_strategy", "Temporal Out-of-Time Holdout") if model_meta else "Temporal Out-of-Time Holdout"
    top_model = model_meta.get("best_model_name", "RandomForestClassifier") if model_meta else "RandomForestClassifier"
    st.markdown(f"""
    <div class="hud-card">
        <div style="font-size:11px;font-weight:700;color:#8B949E;text-transform:uppercase;">Inference Architecture</div>
        <div style="font-size:15px;font-weight:800;color:#39D353;margin-top:2px;">Pre-Event Classifier Suite</div>
        <div style="font-size:12px;color:#C9D1D9;margin-top:4px;">
            Active Engine: <b>{top_model.split('(')[0]}</b><br>
            Validation: <b>{val_strat.split('(')[0].strip()}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="hud-card">
        <div style="font-size:11px;font-weight:700;color:#8B949E;text-transform:uppercase;">Analytical Algorithms</div>
        <div style="font-size:15px;font-weight:800;color:#FFA657;margin-top:2px;">Deterministic & Statistical</div>
        <div style="font-size:12px;color:#C9D1D9;margin-top:4px;">
            Threat Scoring: <b>Log1p 5-Factor Index</b><br>
            Forecasting: <b>Holt's Linear / Holdout Backtest</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 3. Data Flow & Processing Pipeline
st.subheader("2. Platform Data Flow & Pipeline Architecture")
p1, p2, p3, p4 = st.columns(4)
with p1:
    st.markdown("""<div class="flow-step">1. Ingestion & Preprocessing<br><span style="font-size:11px;color:#8B949E;font-weight:400;">Dynamic Schema Adapter, Parquet Cache</span></div>""", unsafe_allow_html=True)
with p2:
    st.markdown("""<div class="flow-step">2. Feature Engineering<br><span style="font-size:11px;color:#8B949E;font-weight:400;">Pre-Event Context (Zero Outcome Leakage)</span></div>""", unsafe_allow_html=True)
with p3:
    st.markdown("""<div class="flow-step">3. Temporal Holdout Split<br><span style="font-size:11px;color:#8B949E;font-weight:400;">Train (<=2011) → Val → Test Holdout</span></div>""", unsafe_allow_html=True)
with p4:
    st.markdown("""<div class="flow-step">4. ML Classification<br><span style="font-size:11px;color:#8B949E;font-weight:400;">Multi-Model Benchmark & Permutation FI</span></div>""", unsafe_allow_html=True)

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

p5, p6, p7, p8 = st.columns(4)
with p5:
    st.markdown("""<div class="flow-step" style="border-top-color:#39D353;">5. Deterministic Risk Engine<br><span style="font-size:11px;color:#8B949E;font-weight:400;">Log1p Composite 0–100 Threat Index</span></div>""", unsafe_allow_html=True)
with p6:
    st.markdown("""<div class="flow-step" style="border-top-color:#39D353;">6. Time-Series Forecasting<br><span style="font-size:11px;color:#8B949E;font-weight:400;">Exponential Smoothing & Holdout Backtest</span></div>""", unsafe_allow_html=True)
with p7:
    st.markdown("""<div class="flow-step" style="border-top-color:#39D353;">7. Anomaly Surveillance<br><span style="font-size:11px;color:#8B949E;font-weight:400;">8-Yr Rolling Window Z-Scores (Z ≥ 2.0)</span></div>""", unsafe_allow_html=True)
with p8:
    st.markdown("""<div class="flow-step" style="border-top-color:#39D353;">8. Decision Support Briefing<br><span style="font-size:11px;color:#8B949E;font-weight:400;">13-Section Deterministic Intelligence Brief</span></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 4. Mathematical Formulations & Component Normalization
st.subheader("3. Deterministic Threat Risk Index Formulation")

st.markdown(f"""
<div class="formula-card">
    <div style="font-size:14px;font-weight:800;color:#58A6FF;text-transform:uppercase;margin-bottom:6px;">
        Mathematical Risk Index Formulation (0–100 Scale)
    </div>
    $$\\text{{Threat Index}} = {WEIGHT_FREQUENCY} \\cdot \\tilde{{F}}_{{\\text{{freq}}}} + {WEIGHT_FATALITY} \\cdot \\tilde{{K}}_{{\\text{{fatality}}}} + {WEIGHT_INJURY} \\cdot \\tilde{{I}}_{{\\text{{injury}}}} + {WEIGHT_VELOCITY} \\cdot \\tilde{{V}}_{{\\text{{velocity}}}} + {WEIGHT_DIVERSITY} \\cdot \\tilde{{D}}_{{\\text{{diversity}}}}$$
    <div style="font-size:13px;color:#C9D1D9;margin-top:8px;line-height:1.6;">
        • <b>$\\tilde{{F}}_{{\\text{{freq}}}}$ (Incident Frequency, {int(WEIGHT_FREQUENCY*100)}%):</b> Log1p-normalized cumulative historical event count.<br>
        • <b>$\\tilde{{K}}_{{\\text{{fatality}}}}$ (Fatality Severity, {int(WEIGHT_FATALITY*100)}%):</b> Log1p-normalized total deaths.<br>
        • <b>$\\tilde{{I}}_{{\\text{{injury}}}}$ (Injury Burden, {int(WEIGHT_INJURY*100)}%):</b> Log1p-normalized total non-fatal casualties.<br>
        • <b>$\\tilde{{V}}_{{\\text{{velocity}}}}$ (Recent Velocity, {int(WEIGHT_VELOCITY*100)}%):</b> Event density concentration across the most recent 3 active dataset years.<br>
        • <b>$\\tilde{{D}}_{{\\text{{diversity}}}}$ (Tactical Diversity, {int(WEIGHT_DIVERSITY*100)}%):</b> Min-Max normalized count of distinct tactical methodologies $[1, 9]$.<br>
        • <b>Log1p Min-Max Scaling Formula:</b> $\\tilde{{X}} = \\frac{{\\ln(1+X) - \\ln(1+X_{{\\min}})}}{{\\ln(1+X_{{\\max}}) - \\ln(1+X_{{\\min}})}} \\times 100$. Eliminates score saturation from high-volume conflict zones while maintaining sensitivity across moderate-threat theaters.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 5. Technical Specifications & Methodology Modules
st.subheader("4. Technical Subsystem Specifications")

with st.expander("🔹 1. Data Leakage Prevention & Pre-Event Boundary Standards", expanded=True):
    st.markdown("""
    * **Feature Boundary Rule:** Input features to the classification pipeline are strictly limited to contextual attributes known *prior to or at the onset* of an incident (`region`, `target_type`, `weapon_type`, `suicide`).
    * **Consequence Variable Exclusion:** Post-incident consequence metrics (e.g., casualties `nkill`, injuries `nwound`, perpetrator claiming credit, or structural damage amounts) are strictly isolated from feature space $X$ to prevent post-event outcome leakage.
    * **Preprocessor Isolation:** All categorical encoders and median imputers fit exclusively on the historical training split ($X_{\\text{train}}$) and transform downstream validation/test matrices without lookahead contamination.
    """)

with st.expander("🔹 2. Temporal Out-of-Time Model Validation Protocol", expanded=False):
    st.markdown("""
    * **Temporal Holdout Architecture:** Rather than randomized splitting (which causes future data to leak into past predictions), the dataset uses a chronological 3-way partition:
      * **Training Set:** Historical base period ($\\le 2011$)
      * **Validation Set:** Intermediate tuning period ($2012–2014$)
      * **Holdout Test Set:** Final out-of-sample evaluation period ($2015–2017$)
    * **Cost-Sensitive Class Balancing:** Tree classifiers employ `class_weight="balanced_subsample"`, inversely weighting class loss to handle severe imbalance without synthetic data distortion (SMOTE).
    * **Selection Metric:** Multi-class models are benchmarked primarily on **Macro $F_1$-score** and **Balanced Accuracy** to ensure minority tactical vectors receive equal evaluation weight.
    """)

with st.expander("🔹 3. Longitudinal Time-Series Extrapolation & Backtesting Protocol", expanded=False):
    st.markdown("""
    * **Out-of-Sample Holdout Backtesting:** Forecasting engines (Naive Baseline, Moving Average, SES, Holt's Linear) fit strictly on $t < t_{\\text{holdout}}$ and evaluate forecasts against the isolated final periods.
    * **Performance Metrics:** Evaluated using $\\text{MAE}$, $\\text{RMSE}$, and division-by-zero stabilized $\\text{sMAPE}$:
      $$\\text{sMAPE} = \\frac{100\\%}{n} \\sum_{t=1}^n \\frac{2 |y_t - \\hat{y}_t|}{|y_t| + |\\hat{y}_t| + 1e-6}$$
    * **Analytical Uncertainty Bounds:** Projected horizons calculate empirical $\\pm 1.96\\sigma$ bounds derived from in-sample 1-step residual error variances.
    """)

with st.expander("🔹 4. Statistical Anomaly Surveillance Engine", expanded=False):
    st.markdown("""
    * **Rolling Baseline Formulation:** Evaluates local historical means ($\\mu_t$) and standard deviations ($\\sigma_t$) over a sliding $k$-year window (default $k=8$).
    * **Z-Score Calculation:** $Z_t = \\frac{y_t - \\mu_t}{\\sigma_t}$. Historical periods with $|Z_t| \\ge 2.0$ are flagged as statistically significant surges rather than gradual baseline trend growth.
    * **Severity Stratification:** Outliers are stratified into Mild ($Z \\ge 1.5$), Moderate ($Z \\ge 2.0$), and Critical ($Z \\ge 3.0$) surge events.
    """)

st.markdown("<br>", unsafe_allow_html=True)

# 6. Operational Governance & Disclosures
st.subheader("5. Operational Governance & Scope Disclosures")

st.markdown("""
<div class="hud-card" style="border-left: 4px solid #FFA657;">
    <div style="font-size:13px;font-weight:800;color:#FFA657;text-transform:uppercase;">Analytical Scope & Limitations</div>
    <div style="font-size:13px;color:#E6EDF3;margin-top:6px;line-height:1.6;">
        1. <b>Retrospective Historical Analytics:</b> System scores, forecasts, and classifications reflect retrospective pattern modeling across historical archives. Outputs do not constitute real-time tactical warnings or predictions of specific real-world individuals.<br>
        2. <b>Statistical Non-Causality:</b> Feature importance attributions and sensitivity indices quantify model reliance within historical data; they do not imply direct real-world sociological causation.<br>
        3. <b>Documentation Completeness:</b> Historical reporting completeness varies across historical decades and geographic regions due to evolving documentation standards over time.<br>
        4. <b>Governance Mandate:</b> This platform is engineered strictly for decision support and risk analytics; it must not be used for autonomous operational actions.
    </div>
</div>
""", unsafe_allow_html=True)