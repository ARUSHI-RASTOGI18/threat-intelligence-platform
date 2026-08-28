"""
Module 3: Machine Learning Attack Type Classifier & Counterfactual Decision Support
Location: ./pages/3_Attack_Prediction.py
"""

from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_analytical_data
from src.ml_engine import (
    predict_attack_type,
    load_trained_artifacts,
    simulate_what_if_scenario,
    get_feature_importances,
    find_historical_analogues
)

st.set_page_config(page_title="Attack Prediction | GTI-ARP", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .pred-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.95) 0%, rgba(13, 17, 23, 0.98) 100%);
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .metric-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
    }
    .expl-box {
        background-color: #161B22;
        border-left: 3px solid #58A6FF;
        border-radius: 4px;
        padding: 12px 16px;
        margin-top: 10px;
        font-size: 13px;
        color: #E6EDF3;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization for prediction history
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

model, preprocessor, label_encoder, metadata = load_trained_artifacts()
df = load_analytical_data()

st.title("🤖 Tactical Attack Methodology Classifier & Decision Workspace")
st.caption("Supervised multi-class pattern classification on pre-event contextual signatures with uncertainty quantification and counterfactual simulations.")

if model is None:
    st.error("⚠️ Trained model artifacts not detected. Please run `python train_pipeline.py` in your terminal.")
    st.stop()

# Operational Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Tactical Inference & Explainability",
    "🔬 Counterfactual 'What-If' Simulation",
    "🛡️ Feature Integrity & Leakage Audit",
    "📋 Model Health & Validation Report"
])

# ----------------------------------------------------
# TAB 1: TACTICAL INFERENCE & EXPLAINABILITY
# ----------------------------------------------------
with tab1:
    st.subheader("1. Incident Contextual Input Space")
    
    with st.form("inference_form"):
        c1, c2 = st.columns(2)
        with c1:
            r_in = st.selectbox("Geographic Region", sorted(df["region"].unique().tolist()), index=0)
            t_in = st.selectbox("Target Sector Category", sorted(df["target_type"].unique().tolist()), index=0)
            w_in = st.selectbox("Weapon Category Vector", sorted(df["weapon_type"].unique().tolist()), index=0)
        with c2:
            sui_in = st.selectbox("Suicide Attack Vector?", ["No (0)", "Yes (1)"], index=0)
            suc_in = st.selectbox("Historical Execution State", ["Successful (1)", "Intercepted/Failed (0)"], index=0)
        
        submitted = st.form_submit_button("⚡ Run Tactical Classification Inference", use_container_width=True)

    if submitted:
        input_payload = {
            "region": r_in,
            "target_type": t_in,
            "weapon_type": w_in,
            "suicide": 1 if "Yes" in sui_in else 0,
            "success": 1 if "Successful" in suc_in else 0
        }

        res = predict_attack_type(input_payload)

        # Store in session state prediction history
        st.session_state["prediction_history"].append({
            "Timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Region": r_in,
            "Target": t_in,
            "Weapon": w_in,
            "Suicide": sui_in,
            "Predicted Tactic": res["predicted_attack_type"],
            "Confidence": f"{res['confidence_percentage']:.1f}%",
            "Margin (Top-1/2)": f"{res['top2_margin']} pp"
        })

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Output Metrics Grid
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="pred-card">
                <div style="font-size:11px;color:#8B949E;text-transform:uppercase;font-weight:700;">Predicted Methodology</div>
                <div style="font-size:20px;font-weight:800;color:#58A6FF;margin-top:2px;">{res['predicted_attack_type']}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="pred-card">
                <div style="font-size:11px;color:#8B949E;text-transform:uppercase;font-weight:700;">Model Confidence</div>
                <div style="font-size:20px;font-weight:800;color:{res['confidence_color']};margin-top:2px;">{res['confidence_percentage']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="pred-card">
                <div style="font-size:11px;color:#8B949E;text-transform:uppercase;font-weight:700;">Confidence Tier</div>
                <div style="font-size:20px;font-weight:800;color:{res['confidence_color']};margin-top:2px;">{res['confidence_tier']}</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="pred-card">
                <div style="font-size:11px;color:#8B949E;text-transform:uppercase;font-weight:700;">Top-1 vs Top-2 Margin</div>
                <div style="font-size:20px;font-weight:800;color:#F0F6FC;margin-top:2px;">{res['top2_margin']} pp</div>
            </div>
            """, unsafe_allow_html=True)

        # Why Did the Model Make This Prediction?
        st.subheader("2. Why Did the Model Make This Prediction?")
        st.markdown(f"""
        <div class="expl-box">
            <b>Deterministic Model Rationale:</b> The classifier attributed peak probability to <b>{res['predicted_attack_type']}</b> ({res['confidence_percentage']:.1f}%). 
            Primary statistical reliance concentrates on the selected <b>Weapon Vector ({w_in})</b> and <b>Target Sector ({t_in})</b>.<br>
            • <i>Uncertainty Assessment:</i> A probability margin of <b>{res['top2_margin']} percentage points</b> separates the leading prediction from secondary candidates. 
            {'Low ambiguity between classes.' if res['top2_margin'] >= 25 else 'Elevated multi-class competition detected; evaluate probability distribution.'}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Multiclass Probability Distribution")
            df_probs = pd.DataFrame(list(res["class_probabilities"].items()), columns=["Tactic", "Probability (%)"])
            fig_p = px.bar(
                df_probs,
                x="Probability (%)",
                y="Tactic",
                orientation="h",
                template="plotly_dark",
                color="Probability (%)",
                color_continuous_scale="Teal"
            )
            fig_p.update_layout(yaxis=dict(autorange="reversed"), height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_p, use_container_width=True)

        with col_right:
            st.subheader("Global Feature Importance Profile")
            st.caption("Derived from test-set permutation importance. Indicates statistical model reliance, not direct causality.")
            df_fi = get_feature_importances()
            fig_fi = px.bar(
                df_fi,
                x="importance_pct",
                y="root_feature",
                orientation="h",
                template="plotly_dark",
                color="importance_pct",
                color_continuous_scale="Viridis",
                labels={"importance_pct": "Permutation Importance (%)", "root_feature": "Feature"}
            )
            fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_fi, use_container_width=True)

        # Historical Evidence Matching
        st.markdown("---")
        st.subheader("3. Historical Pattern Evidence (Empirical Corpus Match)")
        analogues = find_historical_analogues(df, input_payload)
        
        h_col1, h_col2 = st.columns([1.2, 1.8])
        with h_col1:
            st.markdown(f"""
            - **Matching Strategy:** `{analogues['match_type']}`
            - **Matching Historical Records:** `{analogues['total_matches']:,}` events
            - **Most Frequent Historical Tactic:** `{analogues.get('top_historical_tactic', 'N/A')}` ({analogues.get('top_historical_freq', 0.0)}%)
            """)
            st.caption("Distinguishes actual historical observed frequency from the supervised model's probabilistic estimate.")
        with h_col2:
            if not analogues["distribution"].empty:
                st.dataframe(analogues["distribution"], use_container_width=True, hide_index=True)

    # Session Prediction History Panel
    if st.session_state["prediction_history"]:
        st.markdown("---")
        with st.expander("🕒 Session Prediction History & Audit Log", expanded=False):
            hist_df = pd.DataFrame(st.session_state["prediction_history"])
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
            
            c_h1, c_h2 = st.columns([1, 4])
            with c_h1:
                if st.button("Clear History"):
                    st.session_state["prediction_history"] = []
                    st.rerun()
            with c_h2:
                csv_hist = hist_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Export History (CSV)", csv_hist, "prediction_audit_log.csv", "text/csv")

# ----------------------------------------------------
# TAB 2: COUNTERFACTUAL WHAT-IF SIMULATION
# ----------------------------------------------------
with tab2:
    st.subheader("Counterfactual 'What-If' Tactical Simulation")
    st.markdown("Isolate the model's behavioral response by contrasting a baseline context against an alternative configuration.")

    col_base, col_mod = st.columns(2)
    with col_base:
        st.markdown("### 🔹 Baseline Context")
        b_r = st.selectbox("Baseline Region", sorted(df["region"].unique().tolist()), key="b_reg")
        b_t = st.selectbox("Baseline Target", sorted(df["target_type"].unique().tolist()), key="b_targ")
        b_w = st.selectbox("Baseline Weapon", sorted(df["weapon_type"].unique().tolist()), key="b_weap")
        b_sui = st.selectbox("Baseline Suicide Vector?", ["No (0)", "Yes (1)"], key="b_sui")

    with col_mod:
        st.markdown("### 🔸 Counterfactual Scenario")
        m_r = st.selectbox("Scenario Region", sorted(df["region"].unique().tolist()), index=min(1, len(df["region"].unique())-1), key="m_reg")
        m_t = st.selectbox("Scenario Target", sorted(df["target_type"].unique().tolist()), key="m_targ")
        m_w = st.selectbox("Scenario Weapon", sorted(df["weapon_type"].unique().tolist()), index=min(1, len(df["weapon_type"].unique())-1), key="m_weap")
        m_sui = st.selectbox("Scenario Suicide Vector?", ["No (0)", "Yes (1)"], key="m_sui")

    payload_base = {"region": b_r, "target_type": b_t, "weapon_type": b_w, "suicide": 1 if "Yes" in b_sui else 0, "success": 1}
    payload_mod = {"region": m_r, "target_type": m_t, "weapon_type": m_w, "suicide": 1 if "Yes" in m_sui else 0, "success": 1}

    sim_res = simulate_what_if_scenario(payload_base, payload_mod)

    if "error" not in sim_res:
        st.markdown("---")
        st.subheader("Scenario Comparison Summary")
        
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**Active Contextual Modifications:**")
            for chg in sim_res["input_changes"]:
                st.markdown(f"- {chg}")
        with sc2:
            st.markdown(f"""
            - **Baseline Predicted Tactic:** `{sim_res['baseline_prediction']}` ({sim_res['baseline_confidence']:.1f}%)
            - **Counterfactual Predicted Tactic:** `{sim_res['scenario_prediction']}` ({sim_res['scenario_confidence']:.1f}%)
            """)

        # Horizontal Probability Shift Chart
        st.subheader("Probability Shift Distribution (Percentage Points)")
        df_shifts = sim_res["shift_table"].copy()
        
        fig_shift = px.bar(
            df_shifts,
            x="Probability Shift (pp)",
            y="Attack Methodology",
            orientation="h",
            template="plotly_dark",
            color="Probability Shift (pp)",
            color_continuous_scale="Spectral"
        )
        fig_shift.update_layout(yaxis=dict(autorange="reversed"), height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_shift, use_container_width=True)

        st.dataframe(df_shifts, use_container_width=True, hide_index=True)

# ----------------------------------------------------
# TAB 3: FEATURE INTEGRITY & LEAKAGE AUDIT
# ----------------------------------------------------
with tab3:
    st.subheader("Pre-Event Feature Integrity & Data Leakage Audit")
    st.markdown("""
    To maintain academic validity, features in a pre-incident tactical classification system must be evaluated for **temporal leakage**.
    Variables known strictly *post-event* (e.g., fatality counts, wounded counts, perpetrator claiming credit) must not serve as pre-event predictors.
    """)

    audit_table = pd.DataFrame([
        {
            "Feature Name": "region",
            "Variable Type": "Categorical",
            "Temporal Availability": "Pre-Event",
            "Leakage Risk": "Low (Safe)",
            "Methodological Justification": "Geographic location of deployment is known before onset."
        },
        {
            "Feature Name": "target_type",
            "Variable Type": "Categorical",
            "Temporal Availability": "Pre-Event / Immediate Onset",
            "Leakage Risk": "Low (Safe)",
            "Methodological Justification": "Target institution (e.g., Police, Infrastructure) defines the tactical selection."
        },
        {
            "Feature Name": "weapon_type",
            "Variable Type": "Categorical",
            "Temporal Availability": "Pre-Event Context",
            "Leakage Risk": "Low (Safe)",
            "Methodological Justification": "Weapon armament represents the primary tactical vehicle."
        },
        {
            "Feature Name": "suicide",
            "Variable Type": "Binary (0/1)",
            "Temporal Availability": "Pre-Event Tactic",
            "Leakage Risk": "Low (Safe)",
            "Methodological Justification": "Tactical operational choice planned prior to execution."
        },
        {
            "Feature Name": "success",
            "Variable Type": "Binary (0/1)",
            "Temporal Availability": "At-Event Outcome",
            "Leakage Risk": "Moderate",
            "Methodological Justification": "Execution success is only confirmed during/after event completion. Excluded in strict ex-ante deployments."
        },
        {
            "Feature Name": "nkill / nwound",
            "Variable Type": "Numerical (Casualties)",
            "Temporal Availability": "Post-Event Consequence",
            "Leakage Risk": "High (Excluded)",
            "Methodological Justification": "Strictly excluded from ML classification to prevent post-event outcome leakage."
        }
    ])

    st.dataframe(audit_table, use_container_width=True, hide_index=True)
    st.info("The model pipeline strictly isolates casualty consequences from the feature matrix $X$.")

# ----------------------------------------------------
# TAB 4: MODEL HEALTH & VALIDATION REPORT
# ----------------------------------------------------
with tab4:
    st.subheader("Model Card & Academic Validation Telemetry")
    if metadata and "best_model_metrics" in metadata:
        mets = metadata["best_model_metrics"]
        
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("Test Accuracy", f"{mets.get('accuracy')}%")
        c_m2.metric("Balanced Accuracy", f"{mets.get('balanced_accuracy')}%")
        c_m3.metric("Macro F1-Score", f"{mets.get('macro_f1')}%")
        c_m4.metric("Weighted F1-Score", f"{mets.get('weighted_f1')}%")

        st.markdown(f"""
        - **Algorithm Name:** `{metadata.get('best_model_name')}`
        - **Validation Scheme:** `{metadata.get('validation_strategy')}`
        - **Training Corpus Size:** `{metadata.get('train_samples'):,}` instances
        - **Evaluation Holdout Size:** `{metadata.get('test_samples'):,}` instances
        - **Target Classes ({len(metadata.get('target_classes', []))}):** `{', '.join(metadata.get('target_classes', []))}`
        """)

        with st.expander("View Full Model Metadata JSON"):
            st.json(metadata)
    else:
        st.warning("Model metadata uninitialized.")

# ----------------------------------------------------
# FOOTER: ACADEMIC RESPONSIBLE AI NOTICE
# ----------------------------------------------------
st.markdown("---")
with st.expander("📖 Model Limitations & Responsible Interpretation Notice", expanded=False):
    st.markdown("""
    * **Statistical Nature of Output:** Model confidence scores reflect conditional probability distributions over historical training distributions and do not constitute real-world certainty.
    * **Non-Causal Attribution:** Permutation and feature importances quantify empirical model reliance within the dataset; they do not imply direct real-world causality.
    * **Non-Operational Mandate:** This workspace is built exclusively for retrospective academic research and algorithmic decision support. It must not be deployed for real-time operational targeting or predictive policing.
    """)