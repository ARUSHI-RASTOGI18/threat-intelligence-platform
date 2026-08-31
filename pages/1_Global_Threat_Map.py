"""
Module 1: Geospatial Threat Intelligence Command Center
Location: ./pages/1_Global_Threat_Map.py
"""

import time
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.data_loader import load_analytical_data, audit_dataset_quality
from src.risk_engine import compute_country_risk_index
from src.analytics import compute_spatial_hotspots, compute_country_dossier, calculate_period_trends

st.set_page_config(page_title="Global Threat Map | GTI-ARP", page_icon="🌐", layout="wide")

st.markdown("""
<style>
    .tactical-header {
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
        padding: 12px 16px;
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
    .dossier-card {
        background-color: #161B22;
        border: 1px solid #388BFD;
        border-radius: 8px;
        padding: 16px;
        margin-top: 10px;
    }
    .signal-banner {
        background: linear-gradient(90deg, rgba(14, 68, 41, 0.4) 0%, rgba(22, 27, 34, 0.8) 100%);
        border-left: 4px solid #39D353;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 14px;
        font-size: 13px;
        color: #E6EDF3;
    }
    .empty-state-box {
        background: #161B22;
        border: 1px dashed #F85149;
        border-radius: 8px;
        padding: 30px;
        text-align: center;
        color: #C9D1D9;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# 1. Load Data
df = load_analytical_data()
risk_df = compute_country_risk_index(df)
quality = audit_dataset_quality(df)

min_yr = int(df["year"].min()) if not df.empty and "year" in df.columns else 1970
max_yr = int(df["year"].max()) if not df.empty and "year" in df.columns else 2017
max_fatality_val = int(df["fatalities"].max()) if not df.empty and "fatalities" in df.columns else 500

if "playback_running" not in st.session_state:
    st.session_state["playback_running"] = False

st.markdown("<div class='tactical-header'>🛰️ Geospatial Threat Intelligence Command Center</div>", unsafe_allow_html=True)
st.caption("Context-Aware Dual Visual Encoding: Categorical Identity (Mode 1: All Vectors) & Monochromatic Density Intensity (Mode 2: Filtered Vector).")

# 2. Controls Grid
with st.container():
    f1, f2, f3, f4 = st.columns([1.6, 1.2, 1.2, 1.2])

    with f1:
        selected_years = st.slider(
            "Temporal Horizon",
            min_value=min_yr,
            max_value=max_yr,
            value=(min_yr, max_yr),
            key="horizon_slider_manual"
        )

    with f2:
        region_options = ["All Regions"] + sorted(df["region"].dropna().unique().tolist()) if "region" in df.columns else ["All Regions"]
        selected_region = st.selectbox("Geographic Sector", region_options, index=0)

    with f3:
        attack_options = ["All Attack Types"] + sorted(df["attack_type"].dropna().unique().tolist()) if "attack_type" in df.columns else ["All Attack Types"]
        selected_attack = st.selectbox("Tactical Vector", attack_options, index=0)

    with f4:
        map_style = st.selectbox("Visualization Mode", [
            "Spatial Tactical Intensity",
            "Threat Density Heatmap",
            "Sovereign Incident Choropleth"
        ], index=0)

with st.expander("🎛️ Advanced Tactical Filters & Temporal Playback Controller", expanded=True):
    af1, af2, af3 = st.columns([1.2, 1.2, 1.2])
    with af1:
        target_options = ["All Targets"] + sorted(df["target_type"].dropna().unique().tolist()) if "target_type" in df.columns else ["All Targets"]
        selected_target = st.selectbox("Target Sector Category", target_options, index=0)
    with af2:
        weapon_options = ["All Weapons"] + sorted(df["weapon_type"].dropna().unique().tolist()) if "weapon_type" in df.columns else ["All Weapons"]
        selected_weapon = st.selectbox("Weapon Category Vector", weapon_options, index=0)
    with af3:
        fatality_range = st.slider("Casualty Severity Filter (Deaths/Event)", 0, max_fatality_val, (0, max_fatality_val))

    st.markdown("---")
    pb_col1, pb_col2, pb_col3, pb_col4 = st.columns([1.2, 1.2, 1.5, 2.5])
    with pb_col1:
        play_btn = st.button("▶ Play Timeline", use_container_width=True)
    with pb_col2:
        stop_btn = st.button("⏹ Stop / Reset", use_container_width=True)
    with pb_col3:
        playback_pace = st.select_slider("Step Interval (s)", options=[1, 2, 3, 4, 5], value=3)
    with pb_col4:
        playback_status_slot = st.empty()
        playback_status_slot.caption("Temporal Playback streams through decades year-by-year updating map coordinates and live HUD metrics.")

if play_btn:
    st.session_state["playback_running"] = True

if stop_btn:
    st.session_state["playback_running"] = False
    st.rerun()

# 3. Colors Configuration
ATTACK_COLOR_MAP = {
    "Armed Assault": {"base": "#EF4444", "rgb": (239, 68, 68)},
    "Assassination": {"base": "#A855F7", "rgb": (168, 85, 247)},
    "Bombing/Explosion": {"base": "#F97316", "rgb": (249, 115, 22)},
    "Facility/Infrastructure Attack": {"base": "#06B6D4", "rgb": (6, 182, 212)},
    "Hijacking": {"base": "#EAB308", "rgb": (234, 179, 8)},
    "Hostage Taking (Kidnapping)": {"base": "#EC4899", "rgb": (236, 72, 153)},
    "Hostage Taking (Barricade Incident)": {"base": "#F43F5E", "rgb": (244, 63, 94)},
    "Unarmed Assault": {"base": "#22C55E", "rgb": (34, 197, 94)},
    "Unknown": {"base": "#94A3B8", "rgb": (148, 163, 184)}
}

DEFAULT_RGB = (148, 163, 184)

def get_intensity_rgba(attack_type: str, intensity: float) -> str:
    cfg = ATTACK_COLOR_MAP.get(attack_type, {"rgb": DEFAULT_RGB})
    r, g, b = cfg["rgb"]
    norm = np.clip(intensity / 100.0, 0.0, 1.0)
    alpha = float(0.28 + (0.72 * (norm ** 1.35)))
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"

def make_monochromatic_colorscale(attack_type: str):
    cfg = ATTACK_COLOR_MAP.get(attack_type, {"rgb": DEFAULT_RGB})
    r, g, b = cfg["rgb"]
    return [
        [0.0, f"rgba({r}, {g}, {b}, 0.25)"],
        [0.25, f"rgba({r}, {g}, {b}, 0.45)"],
        [0.50, f"rgba({r}, {g}, {b}, 0.70)"],
        [0.75, f"rgba({r}, {g}, {b}, 0.88)"],
        [1.0, f"rgba({r}, {g}, {b}, 1.0)"]
    ]

# 4. Render Engine
def render_command_dashboard(active_start_yr: int, active_end_yr: int, is_playback: bool = False):
    df_map = df[(df["year"] >= active_start_yr) & (df["year"] <= active_end_yr)].copy()

    if selected_region != "All Regions" and "region" in df_map.columns:
        df_map = df_map[df_map["region"] == selected_region]
    if selected_attack != "All Attack Types" and "attack_type" in df_map.columns:
        df_map = df_map[df_map["attack_type"] == selected_attack]
    if selected_target != "All Targets" and "target_type" in df_map.columns:
        df_map = df_map[df_map["target_type"] == selected_target]
    if selected_weapon != "All Weapons" and "weapon_type" in df_map.columns:
        df_map = df_map[df_map["weapon_type"] == selected_weapon]

    if "fatalities" in df_map.columns:
        df_map = df_map[(df_map["fatalities"] >= fatality_range[0]) & (df_map["fatalities"] <= fatality_range[1])]

    df_map_geo = df_map.dropna(subset=["latitude", "longitude"]).copy() if ("latitude" in df_map.columns and "longitude" in df_map.columns) else pd.DataFrame()
    total_matches = len(df_map)
    total_geocoded = len(df_map_geo)
    total_fatalities = int(df_map["fatalities"].sum()) if (total_matches > 0 and "fatalities" in df_map.columns) else 0
    avg_deaths_per_inc = (total_fatalities / total_matches) if total_matches > 0 else 0.0
    geocoding_res_pct = ((total_geocoded / total_matches) * 100.0) if total_matches > 0 else 0.0
    active_countries = df_map["country"].nunique() if (total_matches > 0 and "country" in df_map.columns) else 0

    if total_geocoded > 0:
        df_map_geo["grid_lat"] = df_map_geo["latitude"].round(1)
        df_map_geo["grid_lon"] = df_map_geo["longitude"].round(1)

        agg_dict = {"cluster_incidents": ("year", "count")}
        if "fatalities" in df_map_geo.columns: agg_dict["cluster_fatalities"] = ("fatalities", "sum")
        if "injured" in df_map_geo.columns: agg_dict["cluster_injured"] = ("injured", "sum")

        grid_stats = df_map_geo.groupby(["grid_lat", "grid_lon"]).agg(**agg_dict).reset_index()

        fat_comp = np.log1p(grid_stats.get("cluster_fatalities", 0))
        inj_comp = np.log1p(grid_stats.get("cluster_injured", 0))
        grid_stats["raw_cluster_score"] = (
            0.50 * np.log1p(grid_stats["cluster_incidents"]) +
            0.30 * fat_comp +
            0.20 * inj_comp
        )

        min_s = grid_stats["raw_cluster_score"].min()
        max_s = grid_stats["raw_cluster_score"].max()
        denom = max(1e-5, max_s - min_s)

        grid_stats["cluster_intensity"] = ((grid_stats["raw_cluster_score"] - min_s) / denom) * 100.0

        df_map_geo = df_map_geo.merge(
            grid_stats[["grid_lat", "grid_lon", "cluster_incidents", "cluster_intensity"]],
            on=["grid_lat", "grid_lon"],
            how="left"
        )
        df_map_geo["threat_intensity"] = df_map_geo["cluster_intensity"].fillna(0.0).round(1)

        def classify_intensity_tier(val: float) -> str:
            if val >= 80.0: return "Extreme"
            elif val >= 60.0: return "High"
            elif val >= 40.0: return "Moderate"
            elif val >= 20.0: return "Low"
            return "Very Low"

        df_map_geo["intensity_level"] = df_map_geo["threat_intensity"].apply(classify_intensity_tier)

    MAX_POINTS = 6000
    is_sampled = False
    if total_geocoded > MAX_POINTS:
        df_plot = df_map_geo.sample(n=MAX_POINTS, random_state=42)
        is_sampled = True
    else:
        df_plot = df_map_geo

    # Strategic Signal
    if total_matches > 0:
        trends = calculate_period_trends(df_map)
        top_threat_c = df_map["country"].mode()[0] if ("country" in df_map.columns and not df_map["country"].empty) else "N/A"
        dom_vec = df_map["attack_type"].mode()[0] if ("attack_type" in df_map.columns and not df_map["attack_type"].empty) else "N/A"
        yr_counts = df_map.groupby("year").size()
        peak_act_yr = int(yr_counts.idxmax()) if not yr_counts.empty else "N/A"
        peak_act_vol = int(yr_counts.max()) if not yr_counts.empty else 0

        st.markdown(f"""
        <div class="signal-banner">
            <b>🌐 GLOBAL STRATEGIC SIGNAL:</b> Primary Hotspot: <b>{top_threat_c}</b> | 
            Dominant Vector: <b>{dom_vec}</b> | 
            Historical Peak: <b>{peak_act_yr} ({peak_act_vol:,} events)</b> | 
            Trajectory: <b>{trends.get('trend_direction', 'STABLE')} ({trends.get('incident_delta', 0.0):+,.1f}%)</b> | 
            Active Window: <b>{active_start_yr}–{active_end_yr}</b>
        </div>
        """, unsafe_allow_html=True)

    # Executive Metric HUD
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Target Events (Filtered)</div>
            <div class="hud-value">{total_matches:,}</div>
            <div class="hud-sub" style="color:#58A6FF;">{active_countries} Active Sovereignties</div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Cumulative Fatalities</div>
            <div class="hud-value" style="color:#58A6FF;">{total_fatalities:,}</div>
            <div class="hud-sub" style="color:#38bdf8;">{avg_deaths_per_inc:.2f} Deaths / Incident</div>
        </div>
        """, unsafe_allow_html=True)
    with h3:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Geocoding Resolution</div>
            <div class="hud-value" style="color:#39D353;">{geocoding_res_pct:.1f}%</div>
            <div class="hud-sub" style="color:#39D353;">{total_geocoded:,} Coordinate Pairs</div>
        </div>
        """, unsafe_allow_html=True)
    with h4:
        sample_text = f"Sampled ({MAX_POINTS:,} Pts)" if is_sampled else f"100% Full Fidelity ({total_geocoded:,} Pts)"
        playback_status = f"PLAYING (YEAR {active_end_yr})" if is_playback else ("ONLINE" if total_matches > 0 else "NO MATCHES")
        status_color = "#FFA657" if is_playback else ("#39D353" if total_matches > 0 else "#8B949E")
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Stream / Playback Status</div>
            <div class="hud-value" style="color:{status_color}; font-size:18px;">{playback_status}</div>
            <div class="hud-sub" style="color:#8B949E;">{sample_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Map Rendering
    if total_matches == 0:
        st.markdown("""
        <div class="empty-state-box">
            <h4 style="color:#F85149; margin-bottom:8px;">⚠️ No Matching Incidents Found</h4>
            <p style="font-size:13px; margin-bottom:0;">Please broaden your temporal horizon or adjust filters in the control panel.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        if map_style == "Spatial Tactical Intensity":
            fig_map = go.Figure()

            if selected_attack == "All Attack Types":
                for atk_type, grp in df_plot.groupby("attack_type"):
                    marker_colors = [get_intensity_rgba(atk_type, score) for score in grp["threat_intensity"]]
                    marker_sizes = [min(24, max(5, int(np.sqrt(score) * 2.1) + 4)) for score in grp["threat_intensity"]]
                    
                    custom_text = [
                        f"<b>{city}</b>, {country}<br>Year: {yr}<br>Attack: <b>{atk}</b><br>Deaths: <b>{int(k)}</b> | Injured: <b>{int(w)}</b><br>Intensity: <b>{lvl} ({score:.1f}%)</b>"
                        for city, country, yr, atk, k, w, score, lvl in zip(
                            grp.get("city", ["N/A"] * len(grp)), grp.get("country", ["N/A"] * len(grp)), grp["year"],
                            grp["attack_type"], grp.get("fatalities", [0] * len(grp)), grp.get("injured", [0] * len(grp)),
                            grp["threat_intensity"], grp["intensity_level"]
                        )
                    ]

                    fig_map.add_trace(go.Scattergeo(
                        lat=grp["latitude"],
                        lon=grp["longitude"],
                        mode="markers",
                        name=atk_type,
                        hoverinfo="text",
                        hovertext=custom_text,
                        marker=dict(
                            size=marker_sizes,
                            color=marker_colors,
                            line=dict(width=0.5, color="rgba(255,255,255,0.75)")
                        )
                    ))
            else:
                active_scale = make_monochromatic_colorscale(selected_attack)
                marker_sizes = [min(26, max(5, int(np.sqrt(score) * 2.3) + 4)) for score in df_plot["threat_intensity"]]

                custom_text = [
                    f"<b>{city}</b>, {country}<br>Year: {yr}<br>Attack: <b>{atk}</b><br>Deaths: <b>{int(k)}</b> | Injured: <b>{int(w)}</b><br>Intensity: <b>{lvl} ({score:.1f}%)</b>"
                    for city, country, yr, atk, k, w, score, lvl in zip(
                        df_plot.get("city", ["N/A"] * len(df_plot)), df_plot.get("country", ["N/A"] * len(df_plot)), df_plot["year"],
                        df_plot["attack_type"], df_plot.get("fatalities", [0] * len(df_plot)), df_plot.get("injured", [0] * len(df_plot)),
                        df_plot["threat_intensity"], df_plot["intensity_level"]
                    )
                ]

                fig_map.add_trace(go.Scattergeo(
                    lat=df_plot["latitude"],
                    lon=df_plot["longitude"],
                    mode="markers",
                    name=selected_attack,
                    hoverinfo="text",
                    hovertext=custom_text,
                    marker=dict(
                        size=marker_sizes,
                        color=df_plot["threat_intensity"],
                        colorscale=active_scale,
                        cmin=0,
                        cmax=100,
                        showscale=True,
                        line=dict(width=0.6, color="rgba(255,255,255,0.85)")
                    )
                ))

            fig_map.update_geos(
                showcoastlines=True, coastlinecolor="#30363D",
                showland=True, landcolor="#12161E",
                showocean=True, oceancolor="#090C10",
                showlakes=True, lakecolor="#090C10",
                showcountries=True, countrycolor="#30363D",
                projection_type="natural earth",
                bgcolor="#090C10"
            )
            fig_map.update_layout(
                paper_bgcolor="#090C10",
                plot_bgcolor="#090C10",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.12,
                    xanchor="center",
                    x=0.5,
                    font=dict(color="#E6EDF3", size=10),
                    bgcolor="rgba(22, 27, 34, 0.85)"
                ),
                margin=dict(l=0, r=0, t=0, b=40),
                height=600
            )
            st.plotly_chart(fig_map, use_container_width=True, key=f"plotly_geo_{active_start_yr}_{active_end_yr}")

        elif map_style == "Threat Density Heatmap":
            scale_to_use = make_monochromatic_colorscale(selected_attack) if selected_attack != "All Attack Types" else "Viridis"
            fig_map = go.Figure(go.Scattergeo(
                lat=df_plot["latitude"],
                lon=df_plot["longitude"],
                mode="markers",
                marker=dict(
                    size=[min(26, max(6, int(np.sqrt(sc) * 2.2) + 4)) for sc in df_plot["threat_intensity"]],
                    color=df_plot["threat_intensity"],
                    colorscale=scale_to_use,
                    cmin=0,
                    cmax=100,
                    showscale=True,
                    opacity=0.85
                )
            ))
            fig_map.update_geos(showland=True, landcolor="#12161E", showocean=True, oceancolor="#090C10", showcountries=True, countrycolor="#30363D", bgcolor="#090C10")
            fig_map.update_layout(paper_bgcolor="#090C10", plot_bgcolor="#090C10", margin=dict(l=0, r=0, t=0, b=0), height=600)
            st.plotly_chart(fig_map, use_container_width=True, key=f"plotly_heat_{active_start_yr}_{active_end_yr}")

        else:
            country_agg = df_map.groupby("country").size().reset_index(name="Incidents") if "country" in df_map.columns else pd.DataFrame()
            fig_map = go.Figure(go.Choropleth(
                locations=country_agg["country"],
                locationmode="country names",
                z=country_agg["Incidents"],
                colorscale="Viridis",
                marker_line_color="#30363D"
            ))
            fig_map.update_geos(showland=True, landcolor="#12161E", showocean=True, oceancolor="#090C10", bgcolor="#090C10")
            fig_map.update_layout(paper_bgcolor="#090C10", plot_bgcolor="#090C10", margin=dict(l=0, r=0, t=0, b=0), height=600)
            st.plotly_chart(fig_map, use_container_width=True, key=f"plotly_choro_{active_start_yr}_{active_end_yr}")

# 5. Playback Execution
dashboard_placeholder = st.empty()

if st.session_state.get("playback_running", False):
    playback_status_slot.warning(f"▶ **Timeline Playback Streaming:** Stepping every {playback_pace}s from {min_yr} to {max_yr}...")
    
    for curr_playback_yr in range(min_yr, max_yr + 1):
        if not st.session_state.get("playback_running", False):
            break
        
        dashboard_placeholder.empty()
        with dashboard_placeholder.container():
            render_command_dashboard(min_yr, curr_playback_yr, is_playback=True)
        
        time.sleep(playback_pace)
        
    st.session_state["playback_running"] = False
    playback_status_slot.success(f"🏁 **Playback Completed:** Reached year {max_yr}.")
else:
    with dashboard_placeholder.container():
        render_command_dashboard(selected_years[0], selected_years[1], is_playback=False)

# 6. Dossier & Hotspot Grid
st.markdown("---")
grid_left, grid_right = st.columns([1.3, 1.7])

with grid_left:
    st.subheader("📍 Sovereign Intelligence Dossier")
    available_c = sorted(df["country"].unique().tolist()) if "country" in df.columns else []
    
    if available_c:
        sel_dossier_c = st.selectbox("Inspect Sovereign Entity", available_c, index=0)
        try:
            dossier = compute_country_dossier(df, risk_df, sel_dossier_c)
        except TypeError:
            dossier = compute_country_dossier(df, sel_dossier_c)

        if dossier:
            st.markdown(f"""
            <div class="dossier-card">
                <div style="font-size:18px;font-weight:800;color:#58A6FF;">{dossier.get('country', sel_dossier_c)} Tactical Profile</div>
                <div style="font-size:12px;color:#8B949E;margin-bottom:10px;">Corpus: 1970–{max_yr}</div>
                <table style="width:100%;font-size:13px;color:#E6EDF3;line-height:1.7;">
                    <tr><td><b>Historical Threat Score:</b></td><td><code>{dossier.get('threat_score', dossier.get('composite_risk_score', 'N/A'))}/100</code></td></tr>
                    <tr><td><b>Total Logged Events:</b></td><td>{dossier.get('total_incidents', 0):,} incidents</td></tr>
                    <tr><td><b>Casualty Footprint:</b></td><td>{dossier.get('total_fatalities', 0):,} Deaths | {dossier.get('total_injured', 0):,} Injured</td></tr>
                    <tr><td><b>Dominant Tactic:</b></td><td>{dossier.get('dominant_attack', dossier.get('top_attack_type', 'Unknown'))}</td></tr>
                    <tr><td><b>Primary Target Sector:</b></td><td>{dossier.get('dominant_target', dossier.get('top_target_type', 'Unknown'))}</td></tr>
                    <tr><td><b>Historical Peak:</b></td><td>Year <b>{dossier.get('peak_year', 'N/A')}</b></td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

with grid_right:
    st.subheader("🔥 Top Geographic Threat Hotspots")
    st.caption("Spatial density clusters computed via coordinate indexing.")
    
    hotspot_df = compute_spatial_hotspots(df, top_n=5)
    if not hotspot_df.empty:
        st.dataframe(hotspot_df, hide_index=True, use_container_width=True)
    else:
        st.info("No hotspot data to compute for the active dataset.")