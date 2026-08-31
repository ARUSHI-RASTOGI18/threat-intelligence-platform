# 🛡️ Global Threat Intelligence & Analytical Risk Platform (GTI-ARP)
### *Comprehensive Technical Dossier, Empirical Machine Learning Benchmark & Longitudinal Predictive Report*

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit Interface](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Visualization Engine](https://img.shields.io/badge/Plotly-Interactive-3F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Data Pipeline](https://img.shields.io/badge/Data_Engine-Apache_Arrow_Parquet-teal.svg?style=for-the-badge)](https://arrow.apache.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

---

## 📑 Report Navigation Index
1. [Executive Summary & Motivation](#1-executive-summary--project-motivation)
2. [Dataset Provenance & Quality Engineering](#2-dataset-provenance--quality-engineering)
3. [System Architecture & Analytical Taxonomy](#3-system-architecture--analytical-taxonomy)
4. [Machine Learning Experiments & Benchmarks](#4-machine-learning-experiments--benchmarks)
5. [Longitudinal Time-Series Forecasting & Backtesting](#5-longitudinal-time-series-forecasting--backtesting)
6. [Interactive Command HUD & UI Demonstrations](#6-interactive-command-hud--ui-demonstrations)
7. [Empirical Results & Key Findings](#7-empirical-results--key-findings)
8. [Strategic Outcomes & Operational Impact](#8-strategic-outcomes--operational-impact)
9. [Project Limitations & Future Directions](#9-project-limitations--future-directions)
10. [Local Deployment Guide](#10-local-deployment-guide)

---

## 1. Executive Summary & Project Motivation

### 1.1 The Problem Statement
Geopolitical event data is structurally distinct from standard enterprise time-series datasets:
* **Severe Class Imbalance:** Highly lethal incidents (such as Hijackings or Hostage-taking) constitute $< 1\%$ of occurrences, while Bombings and Armed Assaults represent $> 70\%$ of the historical corpus.
* **Temporal Reporting Inhomogeneity:** The rapid proliferation of digital communication and global media post-1998 created an artificial volume surge across records, distorting naive temporal baselines.
* **Retrospective Data Leakage:** Standard random $K$-fold cross-validation mixes future events into past training sets. This yields falsely high accuracy scores (often exceeding 95%) that fail completely in prospective production settings.

### 1.2 Core Platform Objectives
**GTI-ARP** provides a high-reliability, mathematically bounded intelligence platform that:
* Enforces strict **Chronological Out-of-Time Partitioning** to evaluate genuine tactical predictability.
* Decouples **Supervised Machine Learning Classifiers** (probabilistic attack modeling) from **Deterministic Risk Engines** (transparent, closed-form composite scoring).
* Delivers sub-millisecond in-memory analytical query speeds using an optimized **Apache Arrow / Parquet IPC cache**.

---

## 2. Dataset Provenance & Quality Engineering

The analytical core evaluates the **Global Terrorism Database (GTD)** (1970–2017), capturing 181,691 validated records across 205 sovereign territories.

### Data Ingestion & Sanitization Pipeline
```text
Raw Corpus (181,691 Events)
  │
  ├──► Missing Geolocation Processing ────► Regional Centroid Imputation (Lat/Lon)
  ├──► Missing Casualty Resolution ───────► Deterministic Zero-Fatality Assumption
  ├──► Tactical Vector Standardizing ─────► Harmonized 8 Primary Attack Families
  └──► Serialization Pipeline ────────────► High-Throughput Parquet Arrow IPC File
