"""
Module 8: Interactive Data Explorer, Visual Intelligence & Quality Dashboard
Location: ./pages/8_Data_Explorer.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data_loader import load_analytical_data, audit_dataset_quality

st.set_page_config(page_title="Data Explorer | GTI-ARP", page_icon="🔍", layout="wide")

# Custom Dark Command-Center Theme Styling
st.markdown("""
<style>
    .explorer-header {
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
        padding: 10px 14px;
        text-align: center;
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
        font-size: 20px;
        font-weight: 800;
        color: #F0F6FC;
        margin-top: 2px;
    }
    .hud-sub {
        font-size: 11px;
        font-weight: 600;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

df = load_analytical_data()
quality = audit_dataset_quality(df)

st.markdown("<div class='explorer-header'>🔍 Threat Intelligence Data Explorer & Quality Dashboard</div>", unsafe_allow_html=True)
st.caption("Parametric multi-attribute querying, real-time filtered analytical aggregation, and dataset schema integrity auditing.")

tab1, tab2 = st.tabs(["🔍 Query & Visual Intelligence Explorer", "🛡️ Data Quality & Schema Integrity Audit"])

# ----------------------------------------------------
# TAB 1: QUERY & VISUAL INTELLIGENCE EXPLORER
# ----------------------------------------------------
with tab1:
    # 1. Search & Filter Bar
    col1, col2, col3 = st.columns([1.4, 1.3, 1.3])
    with col1:
        search_txt = st.text_input("Text Search (Country, City, Group, Summary)", "", placeholder="e.g. Iraq, Police, Faction...")
    with col2:
        all_attacks = sorted(df["attack_type"].unique().tolist())
        sel_atk = st.multiselect("Filter Attack Methodologies", all_attacks)
    with col3:
        all_targets = sorted(df["target_type"].unique().tolist())
        sel_targ = st.multiselect("Filter Target Sectors", all_targets)

    # 2. Dynamic Filtering Logic (Vectorized & Fast)
    df_q = df.copy()
    
    if search_txt:
        # Check text match across relevant string attributes
        s_clean = search_txt.strip()
        mask = (
            df_q["country"].str.contains(s_clean, case=False, na=False) |
            df_q["city"].str.contains(s_clean, case=False, na=False) |
            df_q["group_name"].str.contains(s_clean, case=False, na=False) |
            df_q["region"].str.contains(s_clean, case=False, na=False)
        )
        if "summary" in df_q.columns:
            mask = mask | df_q["summary"].str.contains(s_clean, case=False, na=False)
        df_q = df_q[mask]

    if sel_atk:
        df_q = df_q[df_q["attack_type"].isin(sel_atk)]

    if sel_targ:
        df_q = df_q[df_q["target_type"].isin(sel_targ)]

    # 3. Compute Metrics on Filtered Data
    total_filtered = len(df_q)
    total_dataset = max(1, len(df))
    filtered_pct = (total_filtered / total_dataset) * 100

    countries_count = df_q["country"].nunique() if total_filtered > 0 else 0
    attacks_count = df_q["attack_type"].nunique() if total_filtered > 0 else 0
    fatalities_sum = int(df_q["fatalities"].sum()) if total_filtered > 0 else 0
    injured_sum = int(df_q["injured"].sum()) if total_filtered > 0 else 0
    
    success_rate = (
        (df_q["success"].sum() / total_filtered * 100) if total_filtered > 0 and "success" in df_q.columns else 0.0
    )

    # 4. Dynamic KPI Summary Cards
    st.markdown("<br>", unsafe_allow_html=True)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    
    with k1:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Total Incidents</div>
            <div class="hud-value" style="color:#58A6FF;">{total_filtered:,}</div>
            <div class="hud-sub" style="color:#8B949E;">{filtered_pct:.1f}% of Corpus</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Countries Covered</div>
            <div class="hud-value">{countries_count}</div>
            <div class="hud-sub" style="color:#58A6FF;">Sovereign Entities</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Attack Types</div>
            <div class="hud-value">{attacks_count}</div>
            <div class="hud-sub" style="color:#8B949E;">Tactical Vectors</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Total Fatalities</div>
            <div class="hud-value" style="color:#F85149;">{fatalities_sum:,}</div>
            <div class="hud-sub" style="color:#FFA657;">{(fatalities_sum/max(1, total_filtered)):.2f} Deaths / Inc</div>
        </div>
        """, unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Total Injured</div>
            <div class="hud-value" style="color:#D29922;">{injured_sum:,}</div>
            <div class="hud-sub" style="color:#D29922;">{(injured_sum/max(1, total_filtered)):.2f} Injured / Inc</div>
        </div>
        """, unsafe_allow_html=True)

    with k6:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Success Rate</div>
            <div class="hud-value" style="color:#39D353;">{success_rate:.1f}%</div>
            <div class="hud-sub" style="color:#39D353;">Execution State</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. Interactive Filtered Intelligence Overview (2x2 Chart Grid)
    if total_filtered > 0:
        st.subheader("📊 Filtered Intelligence Overview")
        st.caption("Visual distribution analytics calculated dynamically from the active query slice.")

        c1, c2 = st.columns(2)
        
        # Chart A: Incidents Over Time
        with c1:
            st.markdown("#### Longitudinal Incident Trajectory")
            ts_data = df_q.groupby("year").size().reset_index(name="Incidents")
            fig_time = px.line(
                ts_data,
                x="year",
                y="Incidents",
                markers=True,
                template="plotly_dark",
                color_discrete_sequence=["#58A6FF"]
            )
            fig_time.update_layout(
                xaxis_title="Calendar Year",
                yaxis_title="Incident Count",
                height=280,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_time, use_container_width=True)

        # Chart B: Attack Type Distribution
        with c2:
            st.markdown("#### Attack Methodology Distribution")
            atk_data = df_q["attack_type"].value_counts().head(8).reset_index()
            atk_data.columns = ["Attack Type", "Incidents"]
            fig_atk = px.bar(
                atk_data,
                x="Incidents",
                y="Attack Type",
                orientation="h",
                template="plotly_dark",
                color="Incidents",
                color_continuous_scale="Teal"
            )
            fig_atk.update_layout(
                yaxis=dict(autorange="reversed"),
                xaxis_title="Incident Count",
                yaxis_title="",
                height=280,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_atk, use_container_width=True)

        c3, c4 = st.columns(2)

        # Chart C: Top 10 Countries by Incidents
        with c3:
            st.markdown("#### Top Affected Sovereign Territories (Top 10)")
            top_countries = df_q["country"].value_counts().head(10).reset_index()
            top_countries.columns = ["Country", "Incidents"]
            fig_c = px.bar(
                top_countries,
                x="Incidents",
                y="Country",
                orientation="h",
                template="plotly_dark",
                color="Incidents",
                color_continuous_scale="Viridis"
            )
            fig_c.update_layout(
                yaxis=dict(autorange="reversed"),
                xaxis_title="Incident Count",
                yaxis_title="",
                height=290,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_c, use_container_width=True)

        # Chart D: Target Type Distribution
        with c4:
            st.markdown("#### Target Sector Allocation")
            targ_data = df_q["target_type"].value_counts().head(7).reset_index()
            targ_data.columns = ["Target Sector", "Count"]
            fig_targ = px.pie(
                targ_data,
                names="Target Sector",
                values="Count",
                template="plotly_dark",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Dark24
            )
            fig_targ.update_layout(
                height=290,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=10))
            )
            st.plotly_chart(fig_targ, use_container_width=True)

    else:
        st.warning("No records match the selected filter configuration.")

    st.markdown("<br>", unsafe_allow_html=True)

    # 6. Detailed Incident Records Table (Paginated & Exportable)
    st.subheader(f"📋 Detailed Incident Records ({total_filtered:,} matches)")
    
    if total_filtered > 0:
        # Pagination controls
        PAGE_SIZE = 50
        total_pages = max(1, (total_filtered + PAGE_SIZE - 1) // PAGE_SIZE)
        
        p_col1, p_col2 = st.columns([1.2, 4])
        with p_col1:
            page_num = st.number_input("Page Selector", min_value=1, max_value=total_pages, value=1, step=1)
        with p_col2:
            st.caption(f"Showing page {page_num} of {total_pages} (Displaying up to {PAGE_SIZE} records per page).")

        start_idx = (page_num - 1) * PAGE_SIZE
        end_idx = start_idx + PAGE_SIZE
        
        # Display slice
        st.dataframe(df_q.iloc[start_idx:end_idx], use_container_width=True)

        # Download button for filtered data
        st.markdown("---")
        csv_data = df_q.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Export Filtered Query Slice ({total_filtered:,} Records) [CSV]",
            data=csv_data,
            file_name="threat_intel_filtered_query.csv",
            mime="text/csv",
            use_container_width=True
        )

# ----------------------------------------------------
# TAB 2: DATA QUALITY & SCHEMA INTEGRITY AUDIT
# ----------------------------------------------------
with tab2:
    st.subheader("Dataset Quality & Schema Integrity Audit")
    st.caption("Deterministic evaluation across Completeness, Spatial Validity, Temporal Consistency, and Uniqueness.")

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Data Quality Index</div>
            <div class="hud-value" style="color:#39D353;">{quality['data_quality_score']}/100</div>
            <div class="hud-sub" style="color:#39D353;">Deterministic Metric</div>
        </div>
        """, unsafe_allow_html=True)

    with q2:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Feature Completeness</div>
            <div class="hud-value" style="color:#58A6FF;">{quality['completeness_pct']}%</div>
            <div class="hud-sub" style="color:#8B949E;">Core Modeling Attributes</div>
        </div>
        """, unsafe_allow_html=True)

    with q3:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Geocoding Integrity</div>
            <div class="hud-value" style="color:#FFA657;">{quality['geocoding_coverage_pct']}%</div>
            <div class="hud-sub" style="color:#FFA657;">Valid Coord Bounds</div>
        </div>
        """, unsafe_allow_html=True)

    with q4:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Repeated Strike Logs</div>
            <div class="hud-value">{quality['duplicate_rows_count']:,}</div>
            <div class="hud-sub" style="color:#8B949E;">{quality['duplicate_rows_pct']}% of Corpus</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Column Null Breakdown and Types Table
    st.markdown("#### Schema Attribute Types & Missing Value Audit")
    null_audit = pd.DataFrame({
        "Attribute Name": df.columns,
        "Data Type": [str(t) for t in df.dtypes],
        "Missing Count": [int(df[c].isnull().sum()) for c in df.columns],
        "Missing Ratio (%)": [(df[c].isnull().sum() / len(df) * 100).round(2) for c in df.columns]
    })
    st.dataframe(null_audit, use_container_width=True, hide_index=True)

    st.markdown("---")
    with st.expander("📖 Data Quality Scoring Methodology & Audit Rules", expanded=False):
        st.markdown("""
        * **Completeness ($35\\%$):** Non-null ratio across critical analytical columns (`year`, `country`, `region`, `attack_type`, `target_type`, `weapon_type`, `fatalities`, `injured`).
        * **Geocoding Validity ($30\\%$):** Ratio of records with non-null latitude/longitude coordinates bounded within valid geographical bounds ($-90 \\le \\text{lat} \\le 90$, $-180 \\le \\text{lon} \\le 180$).
        * **Temporal Consistency ($20\\%$):** Proportion of events falling within valid historical bounds ($1900–2030$).
        * **Uniqueness ($15\\%$):** Evaluation of identical record profiles while preserving legitimate coordinated multi-target attack entries.
        """)