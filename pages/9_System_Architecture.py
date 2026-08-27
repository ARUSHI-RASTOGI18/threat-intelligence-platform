"""
Module 10: System Architecture & Academic Methodology
Location: ./pages/9_System_Architecture.py
"""

import streamlit as st

st.set_page_config(page_title="System Architecture | GTI-ARP", page_icon="🏛️", layout="wide")

st.title("System Architecture & Research Methodology")
st.markdown("End-to-end analytical pipeline, taxonomy of methods, and technological stack.")

st.subheader("1. End-to-End Platform Pipeline")

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
|    STATISTICAL ANALYTICS     | |     PRE-EVENT ML PIPELINE    |
|  * 5-Yr Rolling Z-Score      | |  * Scikit-Learn Pipeline     |
|  * Period Trajectory Deltas  | |  * Random Forest Classifier  |
|  * Natural Earth Geo Maps    | |  * Feature Importances       |
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
|     (Holt's Double Exponential Smoothing | MAE/MAPE)   |
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
st.subheader("2. Academic Taxonomy: Distinguishing AI vs Analytics")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **Machine Learning (Supervised):**
    * **Algorithm:** Random Forest Multiclass Classifier.
    * **Task:** Tactical Attack Type Classification based on pre-event context.
    * **Validation:** Temporal Out-of-Time Holdout Split + Stratified validation.
    * **Explainability:** Scikit-Learn Gini Importance / MDI aggregation.

    **Longitudinal Forecasting:**
    * **Algorithm:** Holt's Linear Double Exponential Smoothing.
    * **Validation:** Out-of-sample backtesting against Naive baseline.
    """)

with col2:
    st.markdown("""
    **Deterministic Analytics:**
    * **Threat Risk Scoring:** Transparent weighted formula (0 - 100).
    * **Scenario Simulation:** Deterministic recalculation of risk components.

    **Statistical Methods:**
    * **Anomaly Detection:** 5-year rolling window Z-scores ($Z \\ge 2.0$).
    * **Trend Velocity:** Two-period percentage change calculations.
    * **Intelligence Synthesis:** Rule-based parametric document generation.
    """)

st.markdown("---")
st.subheader("3. Technology Stack Inventory")
st.markdown("""
* **Core Language:** Python
* **Web Framework:** Streamlit
* **Data Processing:** Pandas, NumPy, PyArrow (Parquet caching)
* **Machine Learning:** Scikit-Learn, Joblib
* **Data Visualization:** Plotly Express & Plotly Graph Objects
""")