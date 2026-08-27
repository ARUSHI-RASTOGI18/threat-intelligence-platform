"""
Module 5: Threat Risk Engine & Scenario Simulator
Location: ./pages/4_Threat_Risk_Engine.py
"""

import streamlit as st
import plotly.express as px
from src.data_loader import load_analytical_data
from src.risk_engine import compute_country_risk_index, simulate_threat_score

st.set_page_config(page_title="Threat Risk Engine | GTI-ARP", page_icon="⚖️", layout="wide")

st.title("Analytical Threat Risk Engine & Scenario Simulator")
st.markdown("Deterministic, transparent mathematical ranking of sovereign risk profiles.")

df = load_analytical_data()
risk_df = compute_country_risk_index(df)

tab1, tab2 = st.tabs(["📊 Sovereign Risk Index Ranking", "🧪 Threat Scenario Simulator"])

with tab1:
    st.subheader("Mathematical Index Formulation")
    st.markdown("""
    $$\\text{Risk Score} = 0.35 \\cdot \\tilde{F}_{\\text{freq}} + 0.30 \\cdot \\tilde{K}_{\\text{fatality}} + 0.15 \\cdot \\tilde{I}_{\\text{injury}} + 0.10 \\cdot \\tilde{V}_{\\text{velocity}} + 0.10 \\cdot \\tilde{D}_{\\text{diversity}}$$
    *Components are log-normalized against the empirical distribution to prevent score saturation.*
    """)

    c1, c2 = st.columns([2, 1])
    with c1:
        top_20 = risk_df.head(20)
        fig_r = px.bar(top_20, x="composite_risk_score", y="country", orientation="h", color="risk_level",
                       color_discrete_map={"Critical": "#F85149", "High": "#D29922", "Moderate": "#58A6FF", "Low": "#3FB950", "Minimal": "#8B949E"},
                       template="plotly_dark")
        fig_r.update_layout(yaxis=dict(autorange="reversed"), height=520, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_r, use_container_width=True)
    with c2:
        sel_c = st.selectbox("Inspect Country Profile", risk_df["country"].tolist())
        c_row = risk_df[risk_df["country"] == sel_c].iloc[0]
        st.metric("Composite Threat Score", f"{c_row['composite_risk_score']} / 100", delta=c_row["risk_level"])
        st.markdown(f"- **Frequency Contribution (35%):** `{c_row['norm_freq']:.1f}/100`")
        st.markdown(f"- **Fatality Impact (30%):** `{c_row['norm_fatality']:.1f}/100`")
        st.markdown(f"- **Injury Impact (15%):** `{c_row['norm_injury']:.1f}/100`")
        st.markdown(f"- **Recent Velocity (10%):** `{c_row['norm_velocity']:.1f}/100`")
        st.markdown(f"- **Tactical Diversity (10%):** `{c_row['norm_diversity']:.1f}/100`")

with tab2:
    st.subheader("Analytical Scenario Simulator")
    st.caption("Simulate how hypothetical shifts in activity or lethality alter the computed threat index.")
    
    s_freq = st.slider("Simulated Incident Frequency Factor", 0.0, 100.0, 50.0)
    s_fat = st.slider("Simulated Fatality Factor", 0.0, 100.0, 40.0)
    s_inj = st.slider("Simulated Injury Factor", 0.0, 100.0, 30.0)
    s_vel = st.slider("Simulated Recent Velocity Factor", 0.0, 100.0, 60.0)
    s_div = st.slider("Simulated Target/Tactic Diversity Factor", 0.0, 100.0, 50.0)

    sim_out = simulate_threat_score(s_freq, s_fat, s_inj, s_vel, s_div)
    st.metric("Simulated Composite Threat Score", f"{sim_out['simulated_score']} / 100", delta=sim_out['simulated_tier'])
    st.json(sim_out["weights"])