"""
Module 8: Interactive Data Explorer & Quality Audit Dashboard
Location: ./pages/7_Data_Explorer.py
"""

import streamlit as st
import pandas as pd
from src.data_loader import load_analytical_data, audit_dataset_quality

st.set_page_config(page_title="Data Explorer | GTI-ARP", page_icon="🔍", layout="wide")

st.title("Threat Intelligence Data Explorer & Quality Dashboard")

df = load_analytical_data()
quality = audit_dataset_quality(df)

tab1, tab2 = st.tabs(["🔍 Query & Extract Data", "🛡️ Data Quality & Integrity Audit"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        search_txt = st.text_input("Text Search (Country, City, Group)", "")
    with col2:
        sel_atk = st.multiselect("Filter Attack Types", sorted(df["attack_type"].unique().tolist()))
    with col3:
        sel_targ = st.multiselect("Filter Target Types", sorted(df["target_type"].unique().tolist()))

    df_q = df.copy()
    if search_txt:
        mask = df_q["country"].str.contains(search_txt, case=False, na=False) | df_q["city"].str.contains(search_txt, case=False, na=False) | df_q["group_name"].str.contains(search_txt, case=False, na=False)
        df_q = df_q[mask]
    if sel_atk:
        df_q = df_q[df_q["attack_type"].isin(sel_atk)]
    if sel_targ:
        df_q = df_q[df_q["target_type"].isin(sel_targ)]

    st.subheader(f"Query Results ({len(df_q):,} records)")
    st.dataframe(df_q.head(500), use_container_width=True)

    csv_data = df_q.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Export Filtered Query Slice (CSV)", csv_data, file_name="threat_query_export.csv", mime="text/csv")

with tab2:
    st.subheader("Dataset Quality & Schema Integrity Audit")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Data Quality Score", f"{quality['data_quality_score']}/100")
    q2.metric("Completeness", f"{quality['completeness_pct']}%")
    q3.metric("Geocoding Density", f"{quality['geocoding_coverage_pct']}%")
    q4.metric("Repeated Strikes", f"{quality['duplicate_rows_count']:,} ({quality['duplicate_rows_pct']}%)")

    st.markdown("""
    **Audit Methodology:**
    * **Completeness (50%):** Valid data ratio across all core modeling fields.
    * **Geocoding (30%):** Latitude/Longitude coordinate availability for spatial mapping.
    * **Non-Duplicate Ratio (20%):** Accounting for repeated strikes while preserving coordinated multi-target attack logs.
    """)