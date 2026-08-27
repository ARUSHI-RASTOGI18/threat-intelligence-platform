"""
Module 4: Machine Learning Attack Type Classifier with What-If Scenarios
Location: ./pages/3_Attack_Prediction.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import load_analytical_data
from src.model_utils import predict_attack_type, load_trained_artifacts, simulate_what_if_scenario, get_feature_importances

st.set_page_config(page_title="Attack Prediction | GTI-ARP", page_icon="🤖", layout="wide")

st.title("Machine Learning Attack Type Classifier & What-If Simulation")
st.markdown("Predictive tactical classification based on **pre-event features** with native explainability and counterfactual analysis.")

model, preprocessor, label_encoder, metadata = load_trained_artifacts()

if model is None:
    st.error("⚠️ Trained model artifacts not detected. Please run `python train_pipeline.py` in your terminal.")
    st.stop()

# Interactive Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Tactical Inference", "🔬 What-If Scenario Simulator", "📋 Model Card & Validation"])

df = load_analytical_data()

with tab1:
    st.subheader("Single Incident Context Input")
    with st.form("pred_form"):
        c1, c2 = st.columns(2)
        with c1:
            r_in = st.selectbox("Geographic Region", sorted(df["region"].unique().tolist()))
            t_in = st.selectbox("Target Sector Category", sorted(df["target_type"].unique().tolist()))
            w_in = st.selectbox("Weapon Category", sorted(df["weapon_type"].unique().tolist()))
        with c2:
            sui_in = st.selectbox("Suicide Attack Vector?", ["No (0)", "Yes (1)"])
            suc_in = st.selectbox("Execution State", ["Successful (1)", "Intercepted/Failed (0)"])
        submitted = st.form_submit_button("Run Classification Inference")

    if submitted:
        payload = {
            "region": r_in, "target_type": t_in, "weapon_type": w_in,
            "suicide": 1 if "Yes" in sui_in else 0, "success": 1 if "Successful" in suc_in else 0
        }
        res = predict_attack_type(payload)
        st.success(f"**Predicted Attack Methodology:** `{res['predicted_attack_type']}`")
        st.metric("Model Confidence", f"{res['confidence_percentage']:.1f}%")

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Class Probability Distribution")
            df_p = pd.DataFrame(list(res["class_probabilities"].items()), columns=["Tactic", "Probability (%)"])
            fig_p = px.bar(df_p, x="Probability (%)", y="Tactic", orientation="h", template="plotly_dark", color="Probability (%)")
            fig_p.update_layout(yaxis=dict(autorange="reversed"), height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_p, use_container_width=True)
        with c4:
            st.subheader("Global Feature Importance")
            st.caption("Feature importance indicates statistical model reliance, not direct causality.")
            df_fi = get_feature_importances()
            fig_fi = px.bar(df_fi, x="importance_pct", y="root_feature", orientation="h", template="plotly_dark", color="importance_pct", color_continuous_scale="Viridis")
            fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_fi, use_container_width=True)

with tab2:
    st.subheader("Counterfactual 'What-If' Scenario Simulator")
    st.markdown("Compare baseline prediction distributions against an alternative tactical configuration.")
    
    col_b, col_m = st.columns(2)
    with col_b:
        st.markdown("**Baseline Context**")
        b_region = st.selectbox("Base Region", sorted(df["region"].unique().tolist()), key="b_r")
        b_target = st.selectbox("Base Target", sorted(df["target_type"].unique().tolist()), key="b_t")
        b_weapon = st.selectbox("Base Weapon", sorted(df["weapon_type"].unique().tolist()), key="b_w")
    with col_m:
        st.markdown("**Counterfactual Scenario**")
        m_region = st.selectbox("Scenario Region", sorted(df["region"].unique().tolist()), key="m_r")
        m_target = st.selectbox("Scenario Target", sorted(df["target_type"].unique().tolist()), key="m_t")
        m_weapon = st.selectbox("Scenario Weapon", sorted(df["weapon_type"].unique().tolist()), key="m_w")

    base_payload = {"region": b_region, "target_type": b_target, "weapon_type": b_weapon, "suicide": 0, "success": 1}
    mod_payload = {"region": m_region, "target_type": m_target, "weapon_type": m_weapon, "suicide": 0, "success": 1}

    sim_res = simulate_what_if_scenario(base_payload, mod_payload)
    if "error" not in sim_res:
        s1, s2 = st.columns(2)
        s1.metric("Baseline Prediction", f"{sim_res['baseline_prediction']} ({sim_res['baseline_confidence']:.1f}%)")
        s2.metric("Scenario Prediction", f"{sim_res['scenario_prediction']} ({sim_res['scenario_confidence']:.1f}%)")
        st.dataframe(sim_res["shift_table"], use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Model Evaluation & Artifact Card")
    st.json(metadata)
    
    cm = metadata.get("confusion_matrix")
    if cm:
        st.subheader("Multiclass Confusion Matrix (Holdout Set)")
        classes = metadata.get("target_classes", [])
        fig_cm = px.imshow(cm, x=classes, y=classes, text_auto=True, template="plotly_dark", color_continuous_scale="Blues")
        fig_cm.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)