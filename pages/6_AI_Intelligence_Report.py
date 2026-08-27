"""
Module 7: Automated Intelligence Briefing Synthesis
Location: ./pages/6_AI_Intelligence_Report.py
"""

import streamlit as st
from src.data_loader import load_analytical_data
from src.risk_engine import compute_country_risk_index
from src.report_generator import generate_intelligence_digest

st.set_page_config(page_title="AI Intelligence Report | GTI-ARP", page_icon="📄", layout="wide")

st.title("Automated Threat Intelligence Synthesis Brief")
st.markdown("Deterministic, rule-grounded analytical digest generated dynamically from computed dataset statistics.")

df = load_analytical_data()
risk_df = compute_country_risk_index(df)

report_md = generate_intelligence_digest(df, risk_df)

c1, c2 = st.columns([3, 1])
with c1:
    st.markdown(report_md)

with c2:
    st.subheader("Export Digest")
    st.download_button("📥 Download Markdown (.MD)", report_md, file_name="GTI_ARP_Intelligence_Brief.md", mime="text/markdown")
    st.download_button("📥 Download Text (.TXT)", report_md, file_name="GTI_ARP_Intelligence_Brief.txt", mime="text/plain")