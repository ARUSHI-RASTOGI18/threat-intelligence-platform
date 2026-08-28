"""
Module 2: Sovereign Country Intelligence & Head-to-Head Comparison with Dynamic AI Insights
Location: ./pages/2_Country_Analysis.py
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.data_loader import load_analytical_data
from src.risk_engine import compute_country_risk_index

st.set_page_config(page_title="Country Intelligence | GTI-ARP", page_icon="📍", layout="wide")

st.markdown("""
<style>
    .insight-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.95) 0%, rgba(13, 17, 23, 0.98) 100%);
        border-left: 4px solid #58A6FF;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
        border-top: 1px solid rgba(56, 139, 253, 0.2);
        border-right: 1px solid rgba(56, 139, 253, 0.2);
        border-bottom: 1px solid rgba(56, 139, 253, 0.2);
    }
    .insight-title {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #58A6FF;
        margin-bottom: 4px;
    }
    .insight-body {
        font-size: 13px;
        color: #E6EDF3;
        line-height: 1.5;
    }
    .highlight-a { color: #58A6FF; font-weight: 700; }
    .highlight-b { color: #F85149; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

df = load_analytical_data()
risk_df = compute_country_risk_index(df)

st.title("Sovereign Territory Intelligence & Comparative Analytics")

mode = st.radio("Analytics Mode", ["Single Country Intelligence Brief", "Head-to-Head Sovereign Comparison"], horizontal=True)
country_list = sorted(df["country"].unique().tolist())

# ----------------------------------------------------
# 1. SINGLE COUNTRY MODE
# ----------------------------------------------------
if mode == "Single Country Intelligence Brief":
    selected_c = st.selectbox("Select Sovereign Territory", country_list, index=country_list.index("Iraq") if "Iraq" in country_list else 0)
    
    min_yr, max_yr = int(df["year"].min()), int(df["year"].max())
    c_years = st.slider("Filter Timeline (Years)", min_yr, max_yr, (min_yr, max_yr), key="single_yr_slider")
    
    df_c = df[(df["country"] == selected_c) & (df["year"] >= c_years[0]) & (df["year"] <= c_years[1])]
    risk_row = risk_df[risk_df["country"] == selected_c]

    score = risk_row.iloc[0]["composite_risk_score"] if not risk_row.empty else "N/A"
    tier = risk_row.iloc[0]["risk_level"] if not risk_row.empty else "N/A"

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Incidents", f"{len(df_c):,}")
    m2.metric("Total Fatalities", f"{int(df_c['fatalities'].sum()):,}")
    m3.metric("Total Injured", f"{int(df_c['injured'].sum()):,}")
    m4.metric("Threat Index", f"{score} / 100")
    m5.metric("Risk Classification", f"{tier}")

    st.markdown("---")
    st.subheader(f"Automated Intelligence Summary: {selected_c}")
    top_targ = df_c["target_type"].mode()[0] if not df_c.empty else "N/A"
    top_atk = df_c["attack_type"].mode()[0] if not df_c.empty else "N/A"
    peak_yr = df_c.groupby("year").size().idxmax() if not df_c.empty else "N/A"

    st.markdown(f"""
    * **Historical Profile:** {selected_c} accounts for **{len(df_c):,}** cataloged incidents and **{int(df_c['fatalities'].sum()):,}** fatalities in the selected window.
    * **Temporal Peak:** Activity concentrated around peak year **{peak_yr}**.
    * **Tactical Allocation:** Dominant methodology is **{top_atk}**, primarily targeted against **{top_targ}**.
    * **Analytical Evaluation:** Assigned a threat score of **{score}/100 ({tier})** based on casualty burden and activity velocity.
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Historical Incident Trajectory")
        yr_c = df_c.groupby("year").size().reset_index(name="Incidents")
        fig_c = px.area(yr_c, x="year", y="Incidents", template="plotly_dark", color_discrete_sequence=["#58A6FF"])
        fig_c.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
        st.plotly_chart(fig_c, use_container_width=True)
    with c2:
        st.subheader("Target Category Breakdown")
        t_counts = df_c["target_type"].value_counts().head(6).reset_index()
        t_counts.columns = ["Target Type", "Count"]
        fig_t = px.bar(t_counts, x="Count", y="Target Type", orientation="h", template="plotly_dark", color="Count", color_continuous_scale="Viridis")
        fig_t.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10), height=300)
        st.plotly_chart(fig_t, use_container_width=True)

    st.download_button(
        label=f"📥 Download {selected_c} Records (CSV)",
        data=df_c.to_csv(index=False).encode("utf-8"),
        file_name=f"threat_intel_{selected_c.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )

# ----------------------------------------------------
# 2. HEAD-TO-HEAD COMPARISON MODE
# ----------------------------------------------------
else:
    col1, col2 = st.columns(2)
    with col1:
        c_a = st.selectbox("Territory A", country_list, index=country_list.index("Iraq") if "Iraq" in country_list else 0)
    with col2:
        c_b = st.selectbox("Territory B", country_list, index=country_list.index("Afghanistan") if "Afghanistan" in country_list else 1)

    st.markdown("### Comparison Filters")
    f_col1, f_col2, f_col3 = st.columns(3)
    
    with f_col1:
        min_yr, max_yr = int(df["year"].min()), int(df["year"].max())
        comp_years = st.slider("Comparison Horizon (Years)", min_yr, max_yr, (min_yr, max_yr), key="comp_yr_slider")
    
    with f_col2:
        metric_choice = st.selectbox("Primary Comparison Metric", ["Incidents", "Fatalities", "Injuries"])
        
    with f_col3:
        all_attacks = sorted(df["attack_type"].unique().tolist())
        selected_atks = st.multiselect("Filter Attack Methodologies", all_attacks, default=all_attacks[:4])

    df_both = df[df["country"].isin([c_a, c_b]) & (df["year"] >= comp_years[0]) & (df["year"] <= comp_years[1])]
    if selected_atks:
        df_both = df_both[df_both["attack_type"].isin(selected_atks)]

    df_a = df_both[df_both["country"] == c_a]
    df_b = df_both[df_both["country"] == c_b]

    risk_a = risk_df[risk_df["country"] == c_a]
    risk_b = risk_df[risk_df["country"] == c_b]
    s_a = risk_a.iloc[0]["composite_risk_score"] if not risk_a.empty else 0.0
    s_b = risk_b.iloc[0]["composite_risk_score"] if not risk_b.empty else 0.0

    st.markdown("---")
    st.subheader("Sovereign Comparative Metric Matrix")
    
    def pct_diff(va, vb):
        if vb == 0:
            return "N/A"
        return f"{((va - vb) / vb) * 100:+,.1f}%"

    inc_a, inc_b = len(df_a), len(df_b)
    fat_a, fat_b = int(df_a['fatalities'].sum()), int(df_b['fatalities'].sum())
    inj_a, inj_b = int(df_a['injured'].sum()), int(df_b['injured'].sum())

    comp_df = pd.DataFrame({
        "Metric": ["Total Incidents (Filtered)", "Fatalities (Filtered)", "Injuries (Filtered)", "Analytical Threat Score", "Risk Tier"],
        c_a: [f"{inc_a:,}", f"{fat_a:,}", f"{inj_a:,}", f"{s_a}/100", f"{risk_a.iloc[0]['risk_level'] if not risk_a.empty else 'N/A'}"],
        c_b: [f"{inc_b:,}", f"{fat_b:,}", f"{inj_b:,}", f"{s_b}/100", f"{risk_b.iloc[0]['risk_level'] if not risk_b.empty else 'N/A'}"],
        f"Delta ({c_a} vs {c_b})": [pct_diff(inc_a, inc_b), pct_diff(fat_a, fat_b), pct_diff(inj_a, inj_b), f"{s_a - s_b:+.1f} pts", "N/A"]
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # ----------------------------------------------------
    # DYNAMIC COMPARATIVE INTELLIGENCE INSIGHTS ENGINE
    # ----------------------------------------------------
    st.markdown("---")
    st.subheader("💡 Strategic Comparative Intelligence & Findings")

    if inc_a > 0 and inc_b > 0:
        # 1. Timeline & Surge Insights
        yr_grp_a = df_a.groupby("year").size()
        yr_grp_b = df_b.groupby("year").size()
        peak_yr_a, peak_val_a = int(yr_grp_a.idxmax()), int(yr_grp_a.max())
        peak_yr_b, peak_val_b = int(yr_grp_b.idxmax()), int(yr_grp_b.max())

        # 2. Tactic Dominance & Divergence
        top_atk_a = df_a["attack_type"].value_counts().head(1)
        top_atk_b = df_b["attack_type"].value_counts().head(1)
        atk_name_a = top_atk_a.index[0] if not top_atk_a.empty else "N/A"
        atk_pct_a = (top_atk_a.values[0] / inc_a * 100) if not top_atk_a.empty else 0.0
        atk_name_b = top_atk_b.index[0] if not top_atk_b.empty else "N/A"
        atk_pct_b = (top_atk_b.values[0] / inc_b * 100) if not top_atk_b.empty else 0.0

        # 3. Target Vulnerability & Sector Share
        top_targ_a = df_a["target_type"].value_counts().head(1)
        top_targ_b = df_b["target_type"].value_counts().head(1)
        targ_name_a = top_targ_a.index[0] if not top_targ_a.empty else "N/A"
        targ_pct_a = (top_targ_a.values[0] / inc_a * 100) if not top_targ_a.empty else 0.0
        targ_name_b = top_targ_b.index[0] if not top_targ_b.empty else "N/A"
        targ_pct_b = (top_targ_b.values[0] / inc_b * 100) if not top_targ_b.empty else 0.0

        # 4. Lethality Ratios
        lethality_a = fat_a / inc_a if inc_a > 0 else 0.0
        lethality_b = fat_b / inc_b if inc_b > 0 else 0.0

        i_col1, i_col2 = st.columns(2)
        with i_col1:
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">📈 Temporal Trajectory & Historical Peak Surges</div>
                <div class="insight-body">
                    • <span class="highlight-a">{c_a}</span> reached its highest recorded escalation in <b>{peak_yr_a}</b> with <b>{peak_val_a:,}</b> events.<br>
                    • <span class="highlight-b">{c_b}</span> experienced its peak intensity in <b>{peak_yr_b}</b> with <b>{peak_val_b:,}</b> events.<br>
                    • <i>Trajectory Analysis:</i> {'Simultaneous multi-theater escalation observed.' if abs(peak_yr_a - peak_yr_b) <= 2 else f'Staggered operational peaks separated by {abs(peak_yr_a - peak_yr_b)} years.'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">🎯 Tactical Methodology & Signature Vectors</div>
                <div class="insight-body">
                    • <span class="highlight-a">{c_a}</span> relies heavily on <b>{atk_name_a}</b> ({atk_pct_a:.1f}% of total volume).<br>
                    • <span class="highlight-b">{c_b}</span> is dominated by <b>{atk_name_b}</b> ({atk_pct_b:.1f}% of total volume).<br>
                    • <i>Methodological Insight:</i> {'Both territories share identical dominant operational tactics.' if atk_name_a == atk_name_b else 'Tactical divergence indicates distinct weapon availability and operational doctrines.'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with i_col2:
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">🏢 Target Vulnerability & Sector Exposure</div>
                <div class="insight-body">
                    • Primary target in <span class="highlight-a">{c_a}</span>: <b>{targ_name_a}</b> ({targ_pct_a:.1f}% concentration).<br>
                    • Primary target in <span class="highlight-b">{c_b}</span>: <b>{targ_name_b}</b> ({targ_pct_b:.1f}% concentration).<br>
                    • <i>Civilian vs Infrastructure Impact:</i> {'Heavy civilian/property exposure detected across both territories.' if 'Private Citizens' in [targ_name_a, targ_name_b] else 'State/security apparatus remains the primary target vector.'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">⚠️ Lethality Burden & Severity Ratio</div>
                <div class="insight-body">
                    • Average deaths per incident in <span class="highlight-a">{c_a}</span>: <b>{lethality_a:.2f}</b> fatalities/event.<br>
                    • Average deaths per incident in <span class="highlight-b">{c_b}</span>: <b>{lethality_b:.2f}</b> fatalities/event.<br>
                    • <i>Severity Verdict:</i> <span class="{'highlight-a' if lethality_a > lethality_b else 'highlight-b'}">{c_a if lethality_a > lethality_b else c_b}</span> demonstrates <b>{abs(lethality_a - lethality_b):.2f}</b> higher casualty lethality per event.
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Select active filter ranges to generate comparative insights.")

    # ----------------------------------------------------
    # COMPARATIVE VISUALIZATIONS
    # ----------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Comparative Visual Analytics")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown(f"#### Longitudinal {metric_choice} Trajectory ({comp_years[0]}–{comp_years[1]})")
        if metric_choice == "Incidents":
            ts_a = df_a.groupby("year").size().reset_index(name="Value")
            ts_b = df_b.groupby("year").size().reset_index(name="Value")
        elif metric_choice == "Fatalities":
            ts_a = df_a.groupby("year")["fatalities"].sum().reset_index(name="Value")
            ts_b = df_b.groupby("year")["fatalities"].sum().reset_index(name="Value")
        else:
            ts_a = df_a.groupby("year")["injured"].sum().reset_index(name="Value")
            ts_b = df_b.groupby("year")["injured"].sum().reset_index(name="Value")

        ts_a["Country"] = c_a
        ts_b["Country"] = c_b
        ts_merged = pd.concat([ts_a, ts_b])

        if ts_merged.empty:
            st.warning("No data available for the selected filters.")
        else:
            fig_traj = px.line(
                ts_merged,
                x="year",
                y="Value",
                color="Country",
                markers=True,
                template="plotly_dark",
                color_discrete_map={c_a: "#58A6FF", c_b: "#F85149"}
            )
            fig_traj.update_layout(
                yaxis_title=metric_choice,
                xaxis_title="Year",
                margin=dict(l=10, r=10, t=20, b=10),
                height=340
            )
            st.plotly_chart(fig_traj, use_container_width=True)

    with g2:
        st.markdown("#### Tactical Methodology Distribution")
        atk_comp = df_both.groupby(["country", "attack_type"]).size().reset_index(name="Incidents")
        
        if atk_comp.empty:
            st.warning("No attack type data available for the selected filters.")
        else:
            fig_atk_comp = px.bar(
                atk_comp,
                x="attack_type",
                y="Incidents",
                color="country",
                barmode="group",
                template="plotly_dark",
                color_discrete_map={c_a: "#58A6FF", c_b: "#F85149"}
            )
            fig_atk_comp.update_layout(
                xaxis_title="Attack Methodology",
                yaxis_title="Incident Count",
                xaxis_tickangle=-30,
                margin=dict(l=10, r=10, t=20, b=10),
                height=340
            )
            st.plotly_chart(fig_atk_comp, use_container_width=True)

    st.markdown("#### Target Profile Distribution (Top 8 Target Sectors)")
    targ_comp = df_both.groupby(["country", "target_type"]).size().reset_index(name="Incidents")
    top_targets = df_both["target_type"].value_counts().head(8).index.tolist()
    targ_comp_filtered = targ_comp[targ_comp["target_type"].isin(top_targets)]

    if not targ_comp_filtered.empty:
        fig_targ_comp = px.bar(
            targ_comp_filtered,
            x="Incidents",
            y="target_type",
            color="country",
            barmode="group",
            orientation="h",
            template="plotly_dark",
            color_discrete_map={c_a: "#58A6FF", c_b: "#F85149"}
        )
        fig_targ_comp.update_layout(
            yaxis=dict(autorange="reversed"),
            yaxis_title="Target Sector",
            xaxis_title="Incident Count",
            margin=dict(l=10, r=10, t=20, b=10),
            height=360
        )
        st.plotly_chart(fig_targ_comp, use_container_width=True)

    st.markdown("---")
    csv_comp = df_both.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Filtered Comparative Dataset ({c_a} vs {c_b}) [CSV]",
        data=csv_comp,
        file_name=f"comparison_{c_a.lower()}_vs_{c_b.lower()}_filtered.csv",
        mime="text/csv"
    )