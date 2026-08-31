<div align="center">

# 🛡️ Global Threat Intelligence & Analytical Risk Platform (GTI-ARP)
### *Enterprise Multi-Model Intelligence, Geospatial Analytics & Time-Series Forecaster*

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Visualization](https://img.shields.io/badge/Plotly-Interactive-3F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Architecture](https://img.shields.io/badge/Data_Engine-Apache_Arrow_Parquet-teal.svg?style=for-the-badge)](https://arrow.apache.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <b>A production-grade, out-of-time validated defense intelligence telemetry suite processing 181,691 verified conflict incidents across 205 sovereign territories.</b>
</p>

[Key Features](#-key-features) • [System Architecture](#-system-architecture--pipeline) • [Machine Learning & Benchmarks](#-multi-model-machine-learning-suite) • [Forecasting Engine](#-longitudinal-forecasting--walk-forward-validation) • [Installation](#-installation--quickstart)

</div>

---

## 📌 Executive Summary & Problem Statement

Geopolitical conflict data is inherently **sparse, high-variance, class-imbalanced, and subject to severe temporal reporting bias** (especially with the rise of global digital reporting post-1998). Standard analytics tools and basic naive ML models suffer from:

1. **Retrospective Data Leakage:** Using standard random train-test k-fold cross-validation mixes past and future states, inflating model accuracy by up to 35%.
2. **Black-Box Obfuscation:** Masking non-deterministic AI hallucinations with factual statistical metrics.
3. **Outlier Skew:** A single mass-casualty incident artificially distorts localized composite risk rankings.

**GTI-ARP solves this** by establishing an end-to-end analytical command center built upon:
* **Chronological Out-of-Time Holdout Partitions** (Zero data leakage).
* **Dual-Track Engine Demarcation:** Mathematical separation between Supervised ML classifiers and Deterministic Closed-Form Risk Algorithms.
* **Sub-linear Logarithmic Hazard Weighting** to normalize casualty variance.

---

## 📸 Command Center Visual Interface

### 1. Geospatial Threat Command Center (Dual Visual Encoding)
* **Mode 1 (All Tactical Vectors):** Categorical multi-color assignment per operational attack vector.
* **Mode 2 (Filtered Vector):** Continuous monochromatic gradient scaling via dynamic spatial density, log-fatalities, and injury weights. Includes automated historical playback from 1970 to 2017.

<div align="center">
  <img src="assets/global_map.png" alt="Geospatial Map UI" width="95%" style="border-radius:8px; border:1px solid #30363D;">
</div>

<br>

### 2. Time-Series Forecasting & Expanding-Window Backtesting
* Out-of-sample forward projections with **$\pm 1.96\sigma$ empirical residual uncertainty bounds**.
* Interactive comparison between **Static Holdout ($k$-periods)** and **Multi-Epoch Walk-Forward Rolling Validation**.

<div align="center">
  <img src="assets/forecasting.png" alt="Forecasting UI" width="95%" style="border-radius:8px; border:1px solid #30363D;">
</div>

<br>

### 3. Multi-Model Performance & Multiclass Diagnostics
* Full benchmark suite evaluating **Random Forest, HistGradientBoosting, L2 Regularized Logistic Regression**, and **Zero-Variance Dummy Baseline**.
* Normalized Multiclass Confusion Matrix isolating minority-class tactical vectors (Assassinations, Hostage Incidents, Hijackings).

<div align="center">
  <img src="assets/model_performance.png" alt="Model Performance UI" width="95%" style="border-radius:8px; border:1px solid #30363D;">
</div>

---

## 🏗️ System Architecture & Pipeline

The GTI-ARP framework is engineered across four modular layers:

```mermaid
flowchart TD
    subgraph L1 [LAYER 1: Ingestion & Quality Assurance]
        A[📂 Global Terrorism Database: 181,691 Events] --> B[🛡️ Schema Adapter & Coordinate Imputation]
        B --> C[⚡ High-Throughput Parquet Arrow IPC Cache]
    end

    subgraph L2 [LAYER 2: Dual Computation Engines]
        C --> D[📐 Deterministic Risk Engine]
        C --> E[🤖 Supervised Multi-Model ML Pipeline]
        
        D --> D1[Composite Threat Index: 0-100]
        D --> D2[5-Yr Rolling Anomaly Filter: Z >= 2.0]
        
        E --> E1[RandomForest Bagging Classifier]
        E --> E2[HistGradientBoosting Classifier]
        E --> E3[Multinomial Logistic Regression L2]
    end

    subgraph L3 [LAYER 3: Simulation & Time-Series Engine]
        D1 & E1 --> F[🎯 Interactive What-If Scenario Simulator]
        C --> G[📈 Longitudinal Forecaster: Holt's Linear SES]
    end

    subgraph L4 [LAYER 4: Decision Support & Cartography]
        F & G & D2 --> H[🖥️ Streamlit Command HUD & Natural Earth GIS]
        H --> I[📑 Deterministic 13-Section AI Intelligence Brief]
    end

    style L1 fill:#161B22,stroke:#58A6FF,stroke-width:1.5px
    style L2 fill:#161B22,stroke:#39D353,stroke-width:1.5px
    style L3 fill:#161B22,stroke:#FFA657,stroke-width:1.5px
    style L4 fill:#161B22,stroke:#BC8CFF,stroke-width:1.5px
