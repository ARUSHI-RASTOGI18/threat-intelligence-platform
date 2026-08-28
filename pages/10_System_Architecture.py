"""
Module 10: System Architecture & Data Flow Pipeline
Location: ./pages/10_System_Architecture.py
"""

import streamlit as st

st.set_page_config(page_title="System Architecture | GTI-ARP", page_icon="🏛️", layout="wide")

st.title("System Architecture & Data Flow Pipeline")

pipeline_diagram = """
+--------------------------------------------------------+
|               RAW HISTORICAL DATASET                   |
|       (181,691 Records | Temporal Span: 1970-2017)     |
+--------------------------+-----------------------------+
                           |
                           v
+--------------------------------------------------------+
|       DYNAMIC SCHEMA ADAPTER & QUALITY AUDIT           |
|   (Duplicate Resolution | Coordinate Imputation)       |
+--------------------------+-----------------------------+
                           |
        +------------------+------------------+
        v                                     v
+------------------------------+ +------------------------------+
|    STATISTICAL ANALYTICS     | |   MULTI-MODEL ML PIPELINE    |
|  * 5-Yr Rolling Z-Score      | |  * Logistic Reg / RF / HGB   |
|  * Period Trajectory Deltas  | |  * Temporal Holdout Split    |
|  * Natural Earth Geo Maps    | |  * Permutation Importance    |
+--------------+---------------+ +--------------+---------------+
               |                                |
               v                                v
+------------------------------+ +------------------------------+
|  DETERMINISTIC RISK ENGINE   | |     WHAT-IF SIMULATOR        |
|  * Composite Threat Index    | |  * Counterfactual Shifts     |
|  * Log-Normal Sub-Factors    | |  * Probability Deltas        |
+--------------+---------------+ +--------------+---------------+
               |                                |
               +------------------+-------------+
                                  |
                                  v
+--------------------------------------------------------+
|     LONGITUDINAL FORECASTING & BACKTESTING ENGINE      |
|     (Holt's Double Exponential Smoothing | MAE/RMSE)   |
+--------------------------+-----------------------------+
                           |
                           v
+--------------------------------------------------------+
|    DETERMINISTIC AI BRIEFING & DECISION SUPPORT        |
|    (Structured 13-Section Markdown/TXT Synthesizer)    |
+--------------------------------------------------------+
"""

st.code(pipeline_diagram, language="text")

st.markdown("---")
st.subheader("Academic Taxonomy: Distinguishing AI vs Analytics")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **Machine Learning (Supervised):**
    * **Models:** Logistic Regression, Random Forest, HistGradientBoosting.
    * **Task:** Tactical Attack Type Classification based on pre-event context.
    * **Validation:** Three-way Temporal Holdout Split (Train/Val/Test).
    * **Explainability:** Permutation Feature Importance.

    **Longitudinal Forecasting:**
    * **Models:** Holt's Linear, SES, Moving Average, Naive Baseline.
    * **Validation:** Out-of-sample backtesting against Naive baseline.
    """)

with col2:
    st.markdown("""
    **Deterministic Analytics:**
    * **Threat Risk Scoring:** Transparent weighted formula ($0 - 100$).
    * **Scenario Simulation:** Deterministic recalculation of risk components.

    **Statistical Methods:**
    * **Anomaly Detection:** 5-year rolling window Z-scores ($Z \\ge 2.0$).
    * **Trend Velocity:** Two-period percentage change calculations.
    * **Intelligence Synthesis:** Rule-based parametric document generation.
    """)