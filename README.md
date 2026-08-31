# 🛡️ Global Threat Intelligence & Analytical Risk Platform (GTI-ARP)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **An enterprise-grade, deterministic intelligence platform that unifies multi-model supervised machine learning, longitudinal time-series forecasting, and geospatial risk telemetry across 180,000+ historical conflict events.**

---

## 📌 Executive Overview

The **Global Threat Intelligence & Analytical Risk Platform (GTI-ARP)** is an analytical command center designed to process, model, and visualize geopolitical incident patterns from 1970 to 2017. Built on high-performance Apache Arrow / Parquet data pipelines, the platform bridges the gap between raw conflict databases and high-level strategic intelligence.

Unlike black-box dashboards, GTI-ARP strictly demarcates **Supervised Machine Learning** (for tactical vector classification) from **Deterministic Risk Analytics** (for transparent composite threat scoring and rolling anomaly detection).

---

## 🖥️ Command Center Interface

<div align="center">
  <img src="assets/global_map.png" alt="Geospatial Threat Map" width="90%">
  <p><i>Figure 1: Geospatial Threat Intelligence Command Center with Dual Visual Encoding (Categorical Identity + Monochromatic Intensity).</i></p>
</div>

<br>

<div align="center">
  <img src="assets/forecasting.png" alt="Forecasting & Backtesting" width="90%">
  <p><i>Figure 2: Longitudinal Incident Forecasting with Holt's Smoothing and Expanding-Window Walk-Forward Validation.</i></p>
</div>

---

## 🏗️ System Architecture & Data Pipeline

The pipeline uses strict chronological partitioning (Train $\le$ 2011, Val 2012–2014, Test 2015–2017) to completely eliminate temporal lookahead data leakage.

```mermaid
flowchart TD
    subgraph L1 [Layer 1: Corpus Ingestion & Data Quality]
        A[📂 GTD Raw Corpus: 181k Records] --> B[🛡️ Schema Harmonizer & Geocode Imputation]
        B --> C[⚡ Parquet Arrow In-Memory IPC Cache]
    end

    subgraph L2 [Layer 2: Dual Computation Engines]
        C --> D[📐 Deterministic Risk Engine]
        C --> E[🤖 Multi-Model ML Classification]
        
        D --> D1[Composite Threat Score: 0-100]
        D --> D2[Rolling 5-Yr Z-Score Anomalies: Z >= 2.0]
        
        E --> E1[Random Forest Bagging]
        E --> E2[HistGradientBoosting]
        E --> E3[L2 Logistic Regression]
    end

    subgraph L3 [Layer 3: Simulation & Time-Series Engine]
        D1 & E1 --> F[🎯 Parametric What-If Simulator]
        C --> G[📈 Longitudinal Forecaster: Holt's Double Exp]
    end

    subgraph L4 [Layer 4: Executive Decision Support]
        F & G & D2 --> H[🖥️ Interactive Streamlit Command Center]
        H --> I[📑 Deterministic 13-Section AI Synthesis Brief]
    end

    style L1 fill:#161B22,stroke:#58A6FF,stroke-width:1.5px
    style L2 fill:#161B22,stroke:#39D353,stroke-width:1.5px
    style L3 fill:#161B22,stroke:#FFA657,stroke-width:1.5px
    style L4 fill:#161B22,stroke:#BC8CFF,stroke-width:1.5px
