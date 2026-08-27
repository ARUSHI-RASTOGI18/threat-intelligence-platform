"""
Module 3: Sovereign Deep-Dive & Comparative Intelligence
Location: ./pages/2_Country_Analysis.py
"""

import streamlit as st
import plotly.express as px
import pandas as pd
from src.data_loader import load_analytical_data
from src.risk_engine import compute_country_risk_index

st.set_page_config(page_title="Country Analysis | GTI-ARP", page_icon="📍", layout="wide")

df = load_analytical_data()
risk_df = compute_country_risk_index(df)

st.title("Sovereign Territory Intelligence & Head-to-Head Comparison")

mode = st.radio("Mode", ["Single Country Intelligence Brief", "Head-to-Head Sovereign Comparison"], horizontal=True)
country_list = sorted(df["country"].unique().tolist())

if mode == "Single Country Intelligence Brief":
    selected_c = st.selectbox("Select Country", country_list, index=country_list.index("Iraq") if "Iraq" in country_list else 0)
    df_c = df[df["country"] == selected_c]
    risk_row = risk_df[risk_df["country"] == selected_c]

    score = risk_row.iloc[0]["composite_risk_score"] if not risk_row.empty else "N/A"
    tier = risk_row.iloc[0]["risk_level"] if not risk_row.empty else "N/A"

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Incidents", f"{len(df_c):,}")
    m2.metric("Fatalities", f"{int(df_c['fatalities'].sum()):,}")
    m3.metric("Injured", f"{int(df_c['injured'].sum()):,}")
    m4.metric("Threat Score", f"{score} / 100")
    m5.metric("Risk Level", f"{tier}")

    st.markdown("---")
    st.subheader(f"Automated Intelligence Summary: {selected_c}")
    top_targ = df_c["target_type"].mode()[0] if not df_c.empty else "N/A"
    top_atk = df_c["attack_type"].mode()[0] if not df_c.empty else "N/A"
    peak_yr = df_c.groupby("year").size().idxmax() if not df_c.empty else "N/A"

    st.markdown(f"""
    * **Historical Profile:** {selected_c} accounts for **{len(df_c):,}** cataloged incidents and **{int(df_c['fatalities'].sum()):,}** fatalities.
    * **Temporal Peak:** Activity concentrated around peak year **{peak_yr}**.
    * **Tactical Allocation:** Dominant methodology is **{top_atk}**, primarily targeted against **{top_targ}**.
    * **Analytical Evaluation:** Assigned a threat score of **{score}/100 ({tier})** based on weighted casualty burden and historical frequency.
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Historical Incident Curve")
        yr_c = df_c.groupby("year").size().reset_index(name="Incidents")
        fig_c = px.area(yr_c, x="year", y="Incidents", template="plotly_dark", color_discrete_sequence=["#58A6FF"])
        fig_c.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=280)
        st.plotly_chart(fig_c, use_container_width=True)
    with c2:
        st.subheader("Target Category Allocation")
        t_counts = df_c["target_type"].value_counts().head(6).reset_index()
        t_counts.columns = ["Target Type", "Count"]
        fig_t = px.bar(t_counts, x="Count", y="Target Type", orientation="h", template="plotly_dark", color="Count", color_continuous_scale="Viridis")
        fig_t.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10), height=280)
        st.plotly_chart(fig_t, use_container_width=True)

else:
    # Head to head
    col1, col2 = st.columns(2)
    with col1:
        c_a = st.selectbox("Territory A", country_list, index=country_list.index("Iraq") if "Iraq" in country_list else 0)
    with col2:
        c_b = st.selectbox("Territory B", country_list, index=country_list.index("Afghanistan") if "Afghanistan" in country_list else 1)

    df_a = df[df["country"] == c_a]
    df_b = df[df["country"] == c_b]

    risk_a = risk_df[risk_df["country"] == c_a]
    risk_b = risk_df[risk_df["country"] == c_b]
    s_a = risk_a.iloc[0]["composite_risk_score"] if not risk_a.empty else 0.0
    s_b = risk_b.iloc[0]["composite_risk_score"] if not risk_b.empty else 0.0

    st.subheader("Sovereign Comparative Metric Matrix")
    
    def pct_diff(va, vb):
        if vb == 0:
            return "N/A"
        return f"{((va - vb) / vb) * 100:+,.1f}%"

    inc_a, inc_b = len(df_a), len(df_b)
    fat_a, fat_b = int(df_a['fatalities'].sum()), int(df_b['fatalities'].sum())

    comp_df = pd.DataFrame({
        "Metric": ["Total Incidents", "Fatalities", "Injuries", "Analytical Threat Score", "Risk Tier"],
        c_a: [f"{inc_a:,}", f"{fat_a:,}", f"{int(df_a['injured'].sum()):,}", f"{s_a}/100", f"{risk_a.iloc[0]['risk_level'] if not risk_a.empty else 'N/A'}"],
        c_b: [f"{inc_b:,}", f"{fat_b:,}", f"{int(df_b['injured'].sum()):,}", f"{s_b}/100", f"{risk_b.iloc[0]['risk_level'] if not risk_b.empty else 'N/A'}"],
        f"Delta ({c_a} vs {c_b})": [pct_diff(inc_a, inc_b), pct_diff(fat_a, fat_b), pct_diff(df_a['injured'].sum(), df_b['injured'].sum()), f"{s_a - s_b:+.1f} pts", "N/A"]
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.subheader("Comparative Analytical Conclusion")
    higher_c = c_a if s_a > s_b else c_b
    st.info(f"**Comparative Finding:** {higher_c} exhibits a higher historical risk index, driven by greater casualty density and incident volume across recorded periods.")