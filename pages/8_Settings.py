"""
Module 9: System Diagnostics & Model Health
Location: ./pages/8_Settings.py
"""

import streamlit as st
from src.data_loader import load_analytical_data, get_dataset_metadata, audit_dataset_quality
from src.model_utils import load_trained_artifacts

st.set_page_config(page_title="Settings & Health | GTI-ARP", page_icon="⚙️", layout="wide")

st.title("System Diagnostics & Platform Telemetry")

df = load_analytical_data()
meta = get_dataset_metadata(df)
quality = audit_dataset_quality(df)
model, _, _, model_meta = load_trained_artifacts()

h1, h2, h3, h4 = st.columns(4)
h1.metric("Database Health", "ONLINE", delta="Parquet Engine")
h2.metric("ML Classifier", "LOADED" if model else "NOT FOUND")
h3.metric("Data Quality Index", f"{quality['data_quality_score']}%")
h4.metric("Temporal Span", meta['coverage_label'])

st.markdown("---")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Data Pipeline Telemetry")
    st.markdown(f"- **Total Ingested Records:** `{meta['total_records']:,}`")
    st.markdown(f"- **Schema Attributes:** `{meta['total_columns']}` canonical columns")
    st.markdown(f"- **Sovereign Entities:** `{meta['unique_countries']}`")
    st.markdown(f"- **Memory Footprint:** `{quality['memory_mb']} MB`")
    st.markdown(f"- **Geocoding Coordinate Rate:** `{quality['geocoding_coverage_pct']}%`")

with c2:
    st.subheader("Model Artifact Verification")
    if model_meta:
        st.markdown(f"- **Algorithm:** `{model_meta.get('model_name')}`")
        st.markdown(f"- **Validation Strategy:** `{model_meta.get('validation_strategy')}`")
        st.markdown(f"- **Accuracy:** `{model_meta.get('accuracy')}%`")
        st.markdown(f"- **Macro F1:** `{model_meta.get('macro_f1')}%`")
        st.markdown(f"- **Weighted F1:** `{model_meta.get('weighted_f1')}%`")
        st.markdown(f"- **Training Samples:** `{model_meta.get('train_samples'):,}`")
    else:
        st.warning("Model metadata unavailable. Run `python train_pipeline.py`.")

st.markdown("---")
st.subheader("API & Security Configuration")
st.info("Zero External Paid APIs Required. Offline Deterministic Inference Engine Active.")