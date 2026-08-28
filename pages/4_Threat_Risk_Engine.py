"""
Module 4: Analytical Threat Risk Engine & Research-Grade Decision Center
Location: ./pages/4_Threat_Risk_Engine.py
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.data_loader import load_analytical_data
from src.risk_engine import (
    compute_country_risk_index,
    decompose_risk_score,
    compute_country_risk_trajectory,
    compute_risk_sensitivity,
    simulate_threat_score,
    compute_evidence_reliability
)

st.set_page_config(page_title="Threat Risk Engine | GTI-ARP", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    .risk-header {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #58A6FF;
        margin-bottom: 2px;
    }
    .hud-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.95) 0%, rgba(13, 17, 23, 0.98) 100%);
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    }
    .hud-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8B949E;
    }
    .hud-value {
        font-size: 22px;
        font-weight: 800;
        color: #F0F6FC;
        margin-top: 2px;
    }
    .hud-sub {
        font-size: 11px;
        font-weight: 600;
        margin-top: 2px;
    }
    .sim-card-base { background-color: #161B22; border: 1px solid #388BFD; border-radius: 8px; padding: 14px; text-align: center; }
    .sim-card-scenario { background-color: #161B22; border: 1px solid #FFA657; border-radius: 8px; padding: 14px; text-align: center; }
    .sim-card-delta { background-color: #161B22; border: 1px solid #39D353; border-radius: 8px; padding: 14px; text-align: center; }
    .norm-note {
        background-color: #161B22;
        border-left: 3px solid #58A6FF;
        padding: 10px 14px;
        font-size: 12px;
        color: #C9D1D9;
        border-radius: 4px;
        margin-top: 10px;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

df = load_analytical_data()
risk_df = compute_country_risk_index(df)

st.markdown("<div class='risk-header'>⚖️ Analytical Threat Risk Engine & Decision Center</div>", unsafe_allow_html=True)
st.caption("Deterministic mathematical threat scoring, multi-factor Shannon entropy decomposition, longitudinal trajectories, and local sensitivity analysis.")

if risk_df.empty:
    st.error("⚠️ Dataset is empty. Cannot compute sovereign threat indices.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Sovereign Threat Ranking & Decomposition",
    "📈 Historical Trajectory & Transitions",
    "🔬 Factor Sensitivity Analysis",
    "🧪 Enhanced Scenario Simulator & Presets",
    "📖 Mathematical Methodology & Audit"
])

# ----------------------------------------------------
# TAB 1: SOVEREIGN RANKING & DECOMPOSITION
# ----------------------------------------------------
with tab1:
    col_left, col_right = st.columns([1.6, 1.4])

    with col_left:
        st.subheader("1. Global Sovereign Threat Index Ranking")
        top_25 = risk_df.head(25)
        
        fig_r = px.bar(
            top_25,
            x="composite_risk_score",
            y="country",
            orientation="h",
            color="risk_level",
            color_discrete_map={
                "Critical": "#F85149",
                "High": "#FFA657",
                "Moderate": "#58A6FF",
                "Low": "#3FB950",
                "Minimal": "#8B949E"
            },
            template="plotly_dark",
            title="Top 25 Sovereignties by Composite Threat Score"
        )
        fig_r.update_layout(
            yaxis=dict(autorange="reversed"),
            xaxis_title="Composite Threat Index (0–100)",
            yaxis_title="",
            height=580,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with col_right:
        st.subheader("2. Sovereign Profile Inspection & Decomposition")
        country_list = risk_df["country"].tolist()
        sel_c = st.selectbox("Select Sovereign Territory", country_list, index=0)
        
        c_row = risk_df[risk_df["country"] == sel_c].iloc[0]
        df_c = df[df["country"] == sel_c]
        rel_info = compute_evidence_reliability(df_c)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="hud-card">
                <div class="hud-title">Threat Score</div>
                <div class="hud-value" style="color:#58A6FF;">{c_row['composite_risk_score']} <span style="font-size:14px;">/ 100</span></div>
                <div class="hud-sub" style="color:{'#F85149' if c_row['risk_level'] in ['Critical', 'High'] else '#39D353'};">{c_row['risk_level']} Tier</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="hud-card">
                <div class="hud-title">Casualty Volume</div>
                <div class="hud-value" style="color:#F85149;">{int(c_row['total_fatalities']):,}</div>
                <div class="hud-sub" style="color:#FFA657;">{int(c_row['total_injured']):,} Non-Fatal Injured</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="hud-card">
                <div class="hud-title">Data Reliability</div>
                <div class="hud-value" style="color:{rel_info['reliability_color']};">{rel_info['reliability_score']}%</div>
                <div class="hud-sub" style="color:{rel_info['reliability_color']};">{rel_info['reliability_tier']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Mathematical Factor Decomposition")
        decomp_table = decompose_risk_score(c_row)
        
        st.dataframe(
            decomp_table,
            column_config={
                "Risk Component Factor": "Factor",
                "Raw Observed Metric": "Raw Metric",
                "Normalized Score (0-100)": st.column_config.ProgressColumn(
                    "Normalized Risk Component (0–100)",
                    format="%.1f",
                    min_value=0.0,
                    max_value=100.0,
                    help="Scaled relative to the maximum observed value across all sovereign entities in the historical corpus."
                ),
                "Weight": st.column_config.NumberColumn("Weight", format="%.2f"),
                "Contribution": st.column_config.NumberColumn("Contribution", format="%.2f pts")
            },
            hide_index=True,
            use_container_width=True
        )

        st.markdown("""
        <div class="norm-note">
            ℹ️ <b>Methodology & Normalization Disclosure:</b> <code>100.0</code> represents the maximum observed empirical value within the global sovereign universe for that component; it does not represent absolute threat. Tactical Diversity is computed via <b>Normalized Shannon Entropy</b> ($H / \ln K$), evaluating the distribution across tactical vectors rather than raw category counts.
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------
# TAB 2: HISTORICAL TRAJECTORY & TRANSITIONS
# ----------------------------------------------------
with tab2:
    st.subheader("Longitudinal Threat Index Trajectory & Risk-Tier Transitions")
    st.caption("Evaluates historical multi-decade threat progression using identical feature formulation across chronological windows.")

    sel_traj_c = st.selectbox("Inspect Sovereign Territory", risk_df["country"].tolist(), index=0, key="sel_traj")
    traj_data = compute_country_risk_trajectory(df, sel_traj_c)

    if "error" in traj_data:
        st.error(traj_data["error"])
    else:
        df_time = traj_data["timeline"]
        df_trans = traj_data["transitions"]

        t1, t2, t3, t4 = st.columns(4)
        with t1:
            st.metric("Historical Trajectory", traj_data["trajectory_classification"])
        with t2:
            st.metric("Historical Peak", f"Year {traj_data['peak_year']}", f"{traj_data['peak_score']} pts")
        with t3:
            st.metric("Historical Baseline", f"Year {traj_data['min_year']}", f"{traj_data['min_score']} pts")
        with t4:
            st.metric("Overall Horizon Delta", f"{traj_data['overall_delta']:+,.1f} pts")

        fig_traj = px.line(
            df_time,
            x="Year",
            y="Threat Index",
            markers=True,
            template="plotly_dark",
            color_discrete_sequence=["#58A6FF"],
            title=f"Longitudinal Threat Index Curve: {sel_traj_c}"
        )
        if not df_trans.empty:
            for _, tr in df_trans.iterrows():
                fig_traj.add_vline(x=tr["Year"], line_width=1, line_dash="dash", line_color="#FFA657")

        fig_traj.update_layout(
            yaxis=dict(range=[0, 105]),
            xaxis_title="Calendar Year",
            yaxis_title="Threat Score (0–100)",
            height=380,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_traj, use_container_width=True)

        st.markdown("#### Risk-Tier Transition Detector Log")
        if not df_trans.empty:
            st.dataframe(
                df_trans,
                column_config={
                    "Year": "Transition Year",
                    "Previous Tier": "Originating Tier",
                    "New Tier": "Resulting Tier",
                    "Score Delta": "Score Shift (pts)",
                    "Transition Type": "Transition Vector"
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info(f"No risk-tier transitions detected for {sel_traj_c} across historical periods.")

# ----------------------------------------------------
# TAB 3: FACTOR SENSITIVITY ANALYSIS
# ----------------------------------------------------
with tab3:
    st.subheader("Mathematical Factor Sensitivity Matrix")
    st.markdown("Evaluates exact mathematical score response when each component experiences an isolated $+10\%$ perturbation.")

    sel_sens_c = st.selectbox("Inspect Territory Baseline", risk_df["country"].tolist(), index=0, key="sel_sens")
    sens_row = risk_df[risk_df["country"] == sel_sens_c].iloc[0]

    sens_results = compute_risk_sensitivity(
        norm_freq=float(sens_row["norm_freq"]),
        norm_fat=float(sens_row["norm_fatality"]),
        norm_inj=float(sens_row["norm_injury"]),
        norm_vel=float(sens_row["norm_velocity"]),
        norm_div=float(sens_row["norm_diversity"]),
        perturbation_pct=10.0
    )

    st.info(
        f"**Primary Sensitivity Driver:** **{sens_results['most_sensitive_component']}** represents the highest-leverage factor for {sel_sens_c}."
    )

    s_col1, s_col2 = st.columns([1.6, 1.4])
    with s_col1:
        st.dataframe(
            sens_results["sensitivity_table"],
            column_config={
                "Risk Component": "Component Factor",
                "Baseline Value": st.column_config.NumberColumn("Baseline (0-100)", format="%.1f"),
                "Perturbed (+10%)": st.column_config.NumberColumn("Perturbed (+10%)", format="%.1f"),
                "Factor Weight": st.column_config.NumberColumn("Weight", format="%.2f"),
                "New Threat Index": st.column_config.NumberColumn("New Score", format="%.1f"),
                "Absolute Delta (Pts)": st.column_config.NumberColumn("Score Delta", format="+%.2f pts")
            },
            hide_index=True,
            use_container_width=True
        )
    with s_col2:
        fig_sens = px.bar(
            sens_results["sensitivity_table"],
            x="Absolute Delta (Pts)",
            y="Risk Component",
            orientation="h",
            template="plotly_dark",
            color="Absolute Delta (Pts)",
            color_continuous_scale="Viridis",
            title="Score Impact per +10% Factor Shift"
        )
        fig_sens.update_layout(yaxis=dict(autorange="reversed"), height=280, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_sens, use_container_width=True)

# ----------------------------------------------------
# TAB 4: SCENARIO SIMULATOR
# ----------------------------------------------------
with tab4:
    st.subheader("Counterfactual Analytical Scenario Simulator")
    sel_sim_c = st.selectbox("Load Baseline Country Profile", risk_df["country"].tolist(), index=0, key="sel_sim")
    base_row = risk_df[risk_df["country"] == sel_sim_c].iloc[0]

    b_freq = float(base_row["norm_freq"])
    b_fat = float(base_row["norm_fatality"])
    b_inj = float(base_row["norm_injury"])
    b_vel = float(base_row["norm_velocity"])
    b_div = float(base_row["norm_diversity"])
    base_score = float(base_row["composite_risk_score"])
    base_tier = str(base_row["risk_level"])

    preset_choice = st.selectbox("Select Scenario Preset", [
        "Current Baseline (No Change)",
        "Escalation Scenario (+30% Frequency, +20% Velocity)",
        "High-Casualty Surge (+50% Fatality, +30% Injury)",
        "Rapid Operational Velocity Surge (+60% Velocity)",
        "Tactical Diversification (+50% Tactic Diversity)",
        "De-escalation Scenario (-40% Frequency, -40% Velocity)"
    ])

    if preset_choice == "Escalation Scenario (+30% Frequency, +20% Velocity)":
        init_freq, init_fat, init_inj, init_vel, init_div = min(100.0, b_freq * 1.3), b_fat, b_inj, min(100.0, b_vel * 1.2), b_div
    elif preset_choice == "High-Casualty Surge (+50% Fatality, +30% Injury)":
        init_freq, init_fat, init_inj, init_vel, init_div = b_freq, min(100.0, b_fat * 1.5), min(100.0, b_inj * 1.3), b_vel, b_div
    elif preset_choice == "Rapid Operational Velocity Surge (+60% Velocity)":
        init_freq, init_fat, init_inj, init_vel, init_div = b_freq, b_fat, b_inj, min(100.0, b_vel * 1.6), b_div
    elif preset_choice == "Tactical Diversification (+50% Tactic Diversity)":
        init_freq, init_fat, init_inj, init_vel, init_div = b_freq, b_fat, b_inj, b_vel, min(100.0, b_div * 1.5)
    elif preset_choice == "De-escalation Scenario (-40% Frequency, -40% Velocity)":
        init_freq, init_fat, init_inj, init_vel, init_div = max(0.0, b_freq * 0.6), b_fat, b_inj, max(0.0, b_vel * 0.6), b_div
    else:
        init_freq, init_fat, init_inj, init_vel, init_div = b_freq, b_fat, b_inj, b_vel, b_div

    sc_col1, sc_col2 = st.columns(2)
    with sc_col1:
        s_freq = st.slider("Simulated Frequency Score", 0.0, 100.0, float(init_freq), key="sim_f")
        s_fat = st.slider("Simulated Fatality Score", 0.0, 100.0, float(init_fat), key="sim_k")
        s_inj = st.slider("Simulated Injury Score", 0.0, 100.0, float(init_inj), key="sim_i")
    with sc_col2:
        s_vel = st.slider("Simulated Recent Velocity Score", 0.0, 100.0, float(init_vel), key="sim_v")
        s_div = st.slider("Simulated Tactical Diversity Score", 0.0, 100.0, float(init_div), key="sim_d")

    sim_res = simulate_threat_score(s_freq, s_fat, s_inj, s_vel, s_div)
    scenario_score = sim_res["simulated_score"]
    scenario_tier = sim_res["simulated_tier"]

    abs_delta = round(scenario_score - base_score, 1)
    pct_delta = round((abs_delta / max(0.1, base_score)) * 100.0, 1)

    st.markdown("---")
    c_b, c_s, c_d = st.columns(3)
    with c_b:
        st.markdown(f"""
        <div class="sim-card-base">
            <div style="font-size:11px;color:#8B949E;text-transform:uppercase;font-weight:700;">Baseline Index</div>
            <div style="font-size:24px;font-weight:800;color:#58A6FF;margin-top:2px;">{base_score} / 100</div>
            <div style="font-size:12px;font-weight:700;color:#58A6FF;margin-top:2px;">{base_tier} Tier</div>
        </div>
        """, unsafe_allow_html=True)
    with c_s:
        st.markdown(f"""
        <div class="sim-card-scenario">
            <div style="font-size:11px;color:#8B949E;text-transform:uppercase;font-weight:700;">Counterfactual Scenario</div>
            <div style="font-size:24px;font-weight:800;color:#FFA657;margin-top:2px;">{scenario_score} / 100</div>
            <div style="font-size:12px;font-weight:700;color:#FFA657;margin-top:2px;">{scenario_tier} Tier</div>
        </div>
        """, unsafe_allow_html=True)
    with c_d:
        delta_color = "#F85149" if abs_delta > 0 else ("#39D353" if abs_delta < 0 else "#8B949E")
        st.markdown(f"""
        <div class="sim-card-delta">
            <div style="font-size:11px;color:#8B949E;text-transform:uppercase;font-weight:700;">Simulation Shift</div>
            <div style="font-size:24px;font-weight:800;color:{delta_color};margin-top:2px;">{abs_delta:+,.1f} pts</div>
            <div style="font-size:12px;font-weight:700;color:{delta_color};margin-top:2px;">{pct_delta:+,.1f}% Delta</div>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------
# TAB 5: METHODOLOGY
# ----------------------------------------------------
with tab5:
    st.subheader("Mathematical Formulation & Normalization Audit")
    st.markdown("""
    $$\\text{Threat Index} = 0.35 \\cdot \\tilde{F}_{\\text{freq}} + 0.25 \\cdot \\tilde{K}_{\\text{fatality}} + 0.15 \\cdot \\tilde{I}_{\\text{injury}} + 0.15 \\cdot \\tilde{V}_{\\text{velocity}} + 0.10 \\cdot \\tilde{D}_{\\text{diversity}}$$
    
    * **Incident Frequency ($35\\%$):** Log1p scaling normalized to the global historical maximum.
    * **Fatality Severity ($25\\%$):** Cumulative deaths log1p-scaled.
    * **Injury Burden ($15\\%$):** Total non-fatal casualties log1p-scaled.
    * **Recent Velocity ($15\\%$):** Operational density across the most recent 3 active dataset years.
    * **Tactical Diversity ($10\\%$):** Normalized Shannon Entropy $H = -\\sum p_i \\ln(p_i) / \\ln(K)$ measuring operational distribution across tactical vectors.
    """)