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
    .legend-card {
        background: rgba(13, 17, 23, 0.95);
        border: 1px solid rgba(56, 139, 253, 0.3);
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 12px;
    }
    .legend-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        font-weight: 600;
        color: #E6EDF3;
        background: #161B22;
        padding: 4px 10px;
        border-radius: 12px;
        border: 1px solid #30363D;
    }
    .color-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    .gradient-bar {
        height: 10px;
        border-radius: 5px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 6px 0;
    }
</style>
""", unsafe_allow_html=True)

df = load_analytical_data()
risk_df = compute_country_risk_index(df)
quality = audit_dataset_quality(df)

min_yr = int(df["year"].min()) if not df.empty else 1970
max_yr = int(df["year"].max()) if not df.empty else 2017
max_fatality_val = int(df["fatalities"].max()) if not df.empty else 500

if "is_playing" not in st.session_state:
    st.session_state["is_playing"] = False
if "map_year_range" not in st.session_state:
    st.session_state["map_year_range"] = (min_yr, max_yr)
if "playback_current_year" not in st.session_state:
    st.session_state["playback_current_year"] = min_yr

st.markdown("<div class='tactical-header'>🛰️ Geospatial Threat Intelligence Command Center</div>", unsafe_allow_html=True)
st.caption("Context-Aware Dual Visual Encoding: Categorical Identity (Mode 1: All Vectors) & Monochromatic Density Intensity (Mode 2: Filtered Vector).")

# Filter Controls
with st.container():
    f1, f2, f3, f4 = st.columns([1.6, 1.2, 1.2, 1.2])

    with f1:
        selected_years = st.slider(
            "Temporal Horizon",
            min_value=min_yr,
            max_value=max_yr,
            value=st.session_state["map_year_range"],
            key="horizon_slider"
        )
        if not st.session_state["is_playing"]:
            st.session_state["map_year_range"] = selected_years

    with f2:
        region_options = ["All Regions"] + sorted(df["region"].dropna().unique().tolist())
        selected_region = st.selectbox("Geographic Sector", region_options, index=0)

    with f3:
        attack_options = ["All Attack Types"] + sorted(df["attack_type"].dropna().unique().tolist())
        selected_attack = st.selectbox("Tactical Vector", attack_options, index=0)

    with f4:
        map_style = st.selectbox("Visualization Mode", [
            "Spatial Tactical Intensity",
            "Threat Density Heatmap",
            "Sovereign Incident Choropleth"
        ], index=0)

with st.expander("🎛️ Advanced Tactical Filters & Temporal Playback Controller", expanded=False):
    af1, af2, af3 = st.columns([1.2, 1.2, 1.2])
    with af1:
        target_options = ["All Targets"] + sorted(df["target_type"].dropna().unique().tolist())
        selected_target = st.selectbox("Target Sector Category", target_options, index=0)
    with af2:
        weapon_options = ["All Weapons"] + sorted(df["weapon_type"].dropna().unique().tolist())
        selected_weapon = st.selectbox("Weapon Category Vector", weapon_options, index=0)
    with af3:
        fatality_range = st.slider("Casualty Severity Filter (Deaths/Event)", 0, max_fatality_val, (0, max_fatality_val))

    st.markdown("---")
    pb_col1, pb_col2, pb_col3 = st.columns([1.2, 1.2, 3])
    with pb_col1:
        if st.button("▶ Play Timeline", use_container_width=True):
            st.session_state["is_playing"] = True
            st.session_state["playback_current_year"] = min_yr
            st.session_state["map_year_range"] = (min_yr, min_yr)
            st.rerun()

    with pb_col2:
        if st.button("⏹ Stop / Reset", use_container_width=True):
            st.session_state["is_playing"] = False
            st.session_state["playback_current_year"] = min_yr
            st.session_state["map_year_range"] = (min_yr, max_yr)
            st.rerun()

    with pb_col3:
        if st.session_state["is_playing"]:
            st.info(f"▶ **Playback Active:** Advancing through historical records (**Year {st.session_state['map_year_range'][1]}**)...")
        else:
            st.caption("Temporal Playback animates year-by-year across historical decades to reveal conflict escalation.")

# Dynamic Filtering Pipeline
active_min_yr, active_max_yr = st.session_state["map_year_range"]
df_map = df[(df["year"] >= active_min_yr) & (df["year"] <= active_max_yr)]

if selected_region != "All Regions":
    df_map = df_map[df_map["region"] == selected_region]
if selected_attack != "All Attack Types":
    df_map = df_map[df_map["attack_type"] == selected_attack]
if selected_target != "All Targets":
    df_map = df_map[df_map["target_type"] == selected_target]
if selected_weapon != "All Weapons":
    df_map = df_map[df_map["weapon_type"] == selected_weapon]

df_map = df_map[(df_map["fatalities"] >= fatality_range[0]) & (df_map["fatalities"] <= fatality_range[1])]

df_map_geo = df_map.dropna(subset=["latitude", "longitude"]).copy()
total_matches = len(df_map)
total_geocoded = len(df_map_geo)
total_fatalities = int(df_map["fatalities"].sum()) if total_matches > 0 else 0
total_injured = int(df_map["injured"].sum()) if total_matches > 0 else 0
avg_deaths_per_inc = (total_fatalities / total_matches) if total_matches > 0 else 0.0
geocoding_res_pct = ((total_geocoded / total_matches) * 100.0) if total_matches > 0 else 0.0
active_countries = df_map["country"].nunique() if total_matches > 0 else 0

# Dynamic Spatial Threat Intensity Calculation
if total_geocoded > 0:
    df_map_geo["grid_lat"] = df_map_geo["latitude"].round(1)
    df_map_geo["grid_lon"] = df_map_geo["longitude"].round(1)

    grid_stats = df_map_geo.groupby(["grid_lat", "grid_lon"]).agg(
        cluster_incidents=("attack_type", "count"),
        cluster_fatalities=("fatalities", "sum"),
        cluster_injured=("injured", "sum")
    ).reset_index()

    # Dynamic Weighting: 50% Spatial Density + 30% Fatalities + 20% Injuries
    grid_stats["raw_cluster_score"] = (
        0.50 * np.log1p(grid_stats["cluster_incidents"]) +
        0.30 * np.log1p(grid_stats["cluster_fatalities"]) +
        0.20 * np.log1p(grid_stats["cluster_injured"])
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
        if val >= 80.0:
            return "Extreme"
        elif val >= 60.0:
            return "High"
        elif val >= 40.0:
            return "Moderate"
        elif val >= 20.0:
            return "Low"
        return "Very Low"

    df_map_geo["intensity_level"] = df_map_geo["threat_intensity"].apply(classify_intensity_tier)

MAX_POINTS = 7500
is_sampled = False
if total_geocoded > MAX_POINTS:
    df_plot = df_map_geo.sample(n=MAX_POINTS, random_state=42)
    is_sampled = True
else:
    df_plot = df_map_geo

# Strategic Signal Banner
if total_matches > 0:
    trends = calculate_period_trends(df_map)
    top_threat_c = df_map["country"].mode()[0] if not df_map["country"].empty else "N/A"
    dom_vec = df_map["attack_type"].mode()[0] if not df_map["attack_type"].empty else "N/A"
    yr_counts = df_map.groupby("year").size()
    peak_act_yr = int(yr_counts.idxmax()) if not yr_counts.empty else "N/A"
    peak_act_vol = int(yr_counts.max()) if not yr_counts.empty else 0

    st.markdown(f"""
    <div class="signal-banner">
        <b>🌐 GLOBAL STRATEGIC SIGNAL:</b> Primary Hotspot: <b>{top_threat_c}</b> | 
        Dominant Vector: <b>{dom_vec}</b> | 
        Historical Peak Activity: <b>{peak_act_yr} ({peak_act_vol:,} events)</b> | 
        Longitudinal Trajectory: <b>{trends['trend_direction']} ({trends['incident_delta']:+,.1f}%)</b> | 
        Active Window: <b>{active_min_yr}–{active_max_yr}</b>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="signal-banner" style="border-left-color: #FFA657;">
        <b>🌐 GLOBAL STRATEGIC SIGNAL:</b> No incidents match the current filter selection. Adjust parameters to inspect threat telemetry.
    </div>
    """, unsafe_allow_html=True)

# Executive HUD
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
    playback_status = "PLAYING" if st.session_state["is_playing"] else ("ONLINE" if total_matches > 0 else "NO MATCHES")
    status_color = "#FFA657" if st.session_state["is_playing"] else ("#39D353" if total_matches > 0 else "#8B949E")
    st.markdown(f"""
    <div class="hud-card">
        <div class="hud-title">Stream / Playback Status</div>
        <div class="hud-value" style="color:{status_color}; font-size:20px;">{playback_status}</div>
        <div class="hud-sub" style="color:#8B949E;">{sample_text}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Deterministic Categorical Color Mapping
ATTACK_COLOR_MAP = {
    "Armed Assault": {"base": "#EF4444", "name": "Red", "rgb": (239, 68, 68), "ramp": "#fee2e2 0%, #f87171 35%, #dc2626 70%, #991b1b 100%"},
    "Assassination": {"base": "#A855F7", "name": "Purple", "rgb": (168, 85, 247), "ramp": "#f3e8ff 0%, #c084fc 35%, #9333ea 70%, #581c87 100%"},
    "Bombing/Explosion": {"base": "#F97316", "name": "Orange", "rgb": (249, 115, 22), "ramp": "#ffedd5 0%, #fb923c 35%, #ea580c 70%, #7c2d12 100%"},
    "Facility/Infrastructure Attack": {"base": "#06B6D4", "name": "Cyan", "rgb": (6, 182, 212), "ramp": "#cffafe 0%, #22d3ee 35%, #0891b2 70%, #164e63 100%"},
    "Hijacking": {"base": "#EAB308", "name": "Yellow", "rgb": (234, 179, 8), "ramp": "#fef9c3 0%, #facc15 35%, #ca8a04 70%, #713f12 100%"},
    "Hostage Taking (Kidnapping)": {"base": "#EC4899", "name": "Pink", "rgb": (236, 72, 153), "ramp": "#fce7f3 0%, #f472b6 35%, #db2777 70%, #831843 100%"},
    "Hostage Taking (Barricade Incident)": {"base": "#F43F5E", "name": "Rose", "rgb": (244, 63, 94), "ramp": "#ffe4e6 0%, #fb7185 35%, #e11d48 70%, #881337 100%"},
    "Unarmed Assault": {"base": "#22C55E", "name": "Green", "rgb": (34, 197, 94), "ramp": "#dcfce7 0%, #4ade80 35%, #16a34a 70%, #14532d 100%"},
    "Unknown": {"base": "#94A3B8", "name": "Gray", "rgb": (148, 163, 184), "ramp": "#f1f5f9 0%, #94a3b8 35%, #64748b 70%, #1e293b 100%"}
}

DEFAULT_RGB = (148, 163, 184)
DEFAULT_RAMP = "#f1f5f9 0%, #94a3b8 35%, #64748b 70%, #1e293b 100%"

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

# Render Map Canvas
if total_matches == 0:
    st.markdown(f"""
    <div class="empty-state-box">
        <h4 style="color:#F85149; margin-bottom:8px;">⚠️ No Matching Incidents Found</h4>
        <p style="font-size:13px; margin-bottom:0;">
            No records match the active criteria: <b>{selected_region}</b> | <b>{selected_attack}</b> | <b>{selected_target}</b> | <b>{selected_weapon}</b> ({active_min_yr}–{active_max_yr}).<br>
            Please broaden your temporal horizon or adjust filters in the control panel.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    if map_style == "Spatial Tactical Intensity":
        fig_map = go.Figure()

        # SCENARIO A: NO ATTACK FILTER (All vectors chomatic mapping + intensity)
        if selected_attack == "All Attack Types":
            for atk_type, grp in df_plot.groupby("attack_type"):
                marker_colors = [get_intensity_rgba(atk_type, score) for score in grp["threat_intensity"]]
                marker_sizes = [min(24, max(5, int(np.sqrt(score) * 2.1) + 4)) for score in grp["threat_intensity"]]
                
                custom_text = [
                    f"<b>{city}</b>, {country}<br>"
                    f"Year: {yr}<br>"
                    f"Attack Type: <b>{atk}</b><br>"
                    f"Target Type: {targ}<br>"
                    f"Fatalities: <b>{int(k)}</b> | Injured: <b>{int(w)}</b><br>"
                    f"Incident Density: <b>{int(c_inc):,} cluster events</b><br>"
                    f"Intensity Level: <b>{lvl.upper()} ({score:.1f}%)</b>"
                    for city, country, yr, atk, targ, k, w, c_inc, score, lvl in zip(
                        grp["city"], grp["country"], grp["year"],
                        grp["attack_type"], grp["target_type"],
                        grp["fatalities"], grp["injured"],
                        grp["cluster_incidents"], grp["threat_intensity"], grp["intensity_level"]
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

        # SCENARIO B: SPECIFIC ATTACK TYPE SELECTED (Monochromatic single-family intensity)
        else:
            active_scale = make_monochromatic_colorscale(selected_attack)
            marker_sizes = [min(26, max(5, int(np.sqrt(score) * 2.3) + 4)) for score in df_plot["threat_intensity"]]

            custom_text = [
                f"<b>{city}</b>, {country}<br>"
                f"Year: {yr}<br>"
                f"Attack Type: <b>{atk}</b><br>"
                f"Target Type: {targ}<br>"
                f"Fatalities: <b>{int(k)}</b> | Injured: <b>{int(w)}</b><br>"
                f"Incident Density: <b>{int(c_inc):,} cluster events</b><br>"
                f"Intensity Level: <b>{lvl.upper()} ({score:.1f}%)</b>"
                for city, country, yr, atk, targ, k, w, c_inc, score, lvl in zip(
                    df_plot["city"], df_plot["country"], df_plot["year"],
                    df_plot["attack_type"], df_plot["target_type"],
                    df_plot["fatalities"], df_plot["injured"],
                    df_plot["cluster_incidents"], df_plot["threat_intensity"], df_plot["intensity_level"]
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
                    colorbar=dict(
                        title=dict(text=f"{selected_attack}<br>Incident Density", font=dict(color="#58A6FF", size=11)),
                        tickvals=[10, 30, 50, 70, 90],
                        ticktext=["Very Low", "Low", "Moderate", "High", "Extreme"],
                        tickfont=dict(color="#E6EDF3", size=10),
                        bgcolor="rgba(22, 27, 34, 0.85)",
                        bordercolor="rgba(56, 139, 253, 0.3)",
                        borderwidth=1,
                        len=0.75
                    ),
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
                title=dict(text="Tactical Vectors (Color = Type | Intensity = Concentration)", font=dict(color="#58A6FF", size=11)),
                orientation="h",
                yanchor="bottom",
                y=-0.14,
                xanchor="center",
                x=0.5,
                font=dict(color="#E6EDF3", size=10),
                bgcolor="rgba(22, 27, 34, 0.85)",
                bordercolor="rgba(56, 139, 253, 0.3)",
                borderwidth=1
            ),
            margin=dict(l=0, r=0, t=0, b=50),
            height=640
        )
        st.plotly_chart(fig_map, use_container_width=True)

    elif map_style == "Threat Density Heatmap":
        density_agg = df_plot.groupby(["grid_lat", "grid_lon"]).agg(
            density_count=("attack_type", "count"),
            density_fatalities=("fatalities", "sum"),
            density_injured=("injured", "sum"),
            cluster_intensity=("threat_intensity", "mean"),
            primary_country=("country", lambda x: x.mode()[0] if not x.empty else "N/A"),
            primary_city=("city", lambda x: x.mode()[0] if not x.empty else "N/A")
        ).reset_index()

        scale_to_use = make_monochromatic_colorscale(selected_attack) if selected_attack != "All Attack Types" else "Viridis"

        fig_map = go.Figure(go.Scattergeo(
            lat=density_agg["grid_lat"],
            lon=density_agg["grid_lon"],
            mode="markers",
            text=[
                f"<b>{city}</b>, {country}<br>"
                f"Incident Density: <b>{cnt:,} events</b><br>"
                f"Fatalities: <b>{int(fat):,}</b> | Injured: <b>{int(inj):,}</b><br>"
                f"Normalized Intensity: <b>{sc:.1f}/100</b>"
                for city, country, cnt, fat, inj, sc in zip(
                    density_agg["primary_city"], density_agg["primary_country"],
                    density_agg["density_count"], density_agg["density_fatalities"],
                    density_agg["density_injured"], density_agg["cluster_intensity"]
                )
            ],
            hoverinfo="text",
            marker=dict(
                size=[min(28, max(6, int(np.log1p(c) * 4.5) + 4)) for c in density_agg["density_count"]],
                color=density_agg["cluster_intensity"],
                colorscale=scale_to_use,
                cmin=0,
                cmax=100,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Density Intensity", font=dict(color="#58A6FF", size=11)),
                    tickvals=[10, 30, 50, 70, 90],
                    ticktext=["Very Low", "Low", "Moderate", "High", "Extreme"],
                    tickfont=dict(color="#E6EDF3", size=10),
                    bgcolor="rgba(22, 27, 34, 0.85)",
                    bordercolor="rgba(56, 139, 253, 0.3)",
                    borderwidth=1
                ),
                opacity=0.88,
                line=dict(width=0.5, color="rgba(255,255,255,0.4)")
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
        fig_map.update_layout(paper_bgcolor="#090C10", plot_bgcolor="#090C10", margin=dict(l=0, r=0, t=0, b=0), height=640)
        st.plotly_chart(fig_map, use_container_width=True)

    else:
        country_agg = df_map.groupby("country").agg(
            Incidents=("attack_type", "count"),
            Fatalities=("fatalities", "sum"),
            Injuries=("injured", "sum")
        ).reset_index()

        choro_scale = make_monochromatic_colorscale(selected_attack) if selected_attack != "All Attack Types" else "Viridis"

        fig_map = go.Figure(go.Choropleth(
            locations=country_agg["country"],
            locationmode="country names",
            z=country_agg["Incidents"],
            text=country_agg["country"],
            hoverinfo="location+z",
            colorscale=choro_scale,
            marker_line_color="#30363D",
            marker_line_width=0.8,
            colorbar=dict(title=dict(text="Incident Volume", font=dict(color="#58A6FF", size=11)), tickfont=dict(color="#E6EDF3", size=10))
        ))
        fig_map.update_geos(
            showcoastlines=True, coastlinecolor="#30363D",
            showland=True, landcolor="#12161E",
            showocean=True, oceancolor="#090C10",
            showlakes=True, lakecolor="#090C10",
            showcountries=True, countrycolor="rgba(88, 166, 255, 0.3)",
            projection_type="natural earth",
            bgcolor="#090C10"
        )
        fig_map.update_layout(paper_bgcolor="#090C10", plot_bgcolor="#090C10", margin=dict(l=0, r=0, t=0, b=0), height=640)
        st.plotly_chart(fig_map, use_container_width=True)

    # Dynamic Legend Block
    if selected_attack == "All Attack Types":
        pills_html = "".join([
            f'<div class="legend-pill"><span class="color-dot" style="background:{cfg["base"]};"></span>{atk}</div>'
            for atk, cfg in ATTACK_COLOR_MAP.items()
            if atk in df_plot["attack_type"].unique()
        ])

        st.markdown(f"""
        <div class="legend-card">
            <div style="font-size:12px; font-weight:800; color:#58A6FF; text-transform:uppercase; margin-bottom:8px;">
                🎨 Attack Vector Categories (COLOR = Tactical Type)
            </div>
            <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px;">
                {pills_html}
            </div>
            <div style="font-size:11px; font-weight:800; color:#8B949E; text-transform:uppercase; margin-top:8px;">
                ⚡ Threat Intensity Scale (OPACITY & SIZE = Normalized Concentration)
            </div>
            <div class="gradient-bar" style="background: linear-gradient(90deg, rgba(255,255,255,0.25) 0%, rgba(255,255,255,0.45) 25%, rgba(255,255,255,0.70) 50%, rgba(255,255,255,0.88) 75%, rgba(255,255,255,1.0) 100%);"></div>
            <div style="display:flex; justify-content:space-between; font-size:11px; color:#C9D1D9;">
                <span>● Very Low (Sparse area)</span>
                <span>● Low</span>
                <span>● Moderate</span>
                <span>● High</span>
                <span>● Extreme (Severe cluster)</span>
            </div>
            <div style="font-size:11px; color:#8B949E; margin-top:6px;">
                <i>*Intensity represents normalized incident concentration within the active filter.</i>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        cfg = ATTACK_COLOR_MAP.get(selected_attack, {"base": "#58A6FF", "name": "Active", "ramp": DEFAULT_RAMP})
        ramp_css = cfg.get("ramp", DEFAULT_RAMP)

        st.markdown(f"""
        <div class="legend-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <div style="font-size:12px; font-weight:800; color:{cfg['base']}; text-transform:uppercase;">
                    🎯 {selected_attack.upper()} — INCIDENT DENSITY
                </div>
                <div class="legend-pill" style="border-color:{cfg['base']};">
                    <span class="color-dot" style="background:{cfg['base']};"></span>Filtered Tactical Vector
                </div>
            </div>
            <div class="gradient-bar" style="background: linear-gradient(90deg, {ramp_css});"></div>
            <div style="display:flex; justify-content:space-between; font-size:11px; color:#C9D1D9;">
                <span>● Very Low (Faint)</span>
                <span>● Low</span>
                <span>● Moderate</span>
                <span>● High</span>
                <span>● Extreme (Glowing Cluster)</span>
            </div>
            <div style="font-size:11px; color:#8B949E; margin-top:6px;">
                <i>*Intensity represents normalized incident concentration within the active filter ({selected_attack}).</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Command Grid
st.markdown("---")
grid_left, grid_right = st.columns([1.3, 1.7])

with grid_left:
    st.subheader("📍 Sovereign Intelligence Dossier")
    available_c = sorted(df_map["country"].unique().tolist()) if total_matches > 0 else []
    
    if available_c:
        sel_dossier_c = st.selectbox("Inspect Sovereign Entity", available_c, index=0)
        dossier = compute_country_dossier(df_map, risk_df, sel_dossier_c)

        if dossier:
            st.markdown(f"""
            <div class="dossier-card">
                <div style="font-size:18px;font-weight:800;color:#58A6FF;">{dossier['country']} Tactical Profile</div>
                <div style="font-size:12px;color:#8B949E;margin-bottom:10px;">Window: {active_min_yr}–{active_max_yr}</div>
                <table style="width:100%;font-size:13px;color:#E6EDF3;line-height:1.7;">
                    <tr><td><b>Historical Threat Score:</b></td><td><code>{dossier['threat_score']}/100</code> ({dossier['threat_tier']} Tier)</td></tr>
                    <tr><td><b>Observed Trajectory:</b></td><td><b>{dossier['trajectory']}</b> ({dossier['trajectory_delta']:+,.1f}%)</td></tr>
                    <tr><td><b>Total Logged Events:</b></td><td>{dossier['total_incidents']:,} incidents</td></tr>
                    <tr><td><b>Casualty Footprint:</b></td><td>{dossier['total_fatalities']:,} Deaths | {dossier['total_injured']:,} Injured</td></tr>
                    <tr><td><b>Avg Lethality Rate:</b></td><td>{dossier['avg_casualties']} casualties / event</td></tr>
                    <tr><td><b>Dominant Tactic:</b></td><td>{dossier['dominant_attack']}</td></tr>
                    <tr><td><b>Primary Target Sector:</b></td><td>{dossier['dominant_target']}</td></tr>
                    <tr><td><b>Historical Peak:</b></td><td>Year <b>{dossier['peak_year']}</b> ({dossier['peak_year_incidents']:,} events)</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No sovereign records in the active filter selection.")

with grid_right:
    st.subheader("🔥 Top 5 Geographic Threat Hotspots")
    st.caption("Spatial density clusters computed via ~0.5° grid binning with longitudinal velocity status.")
    
    if total_matches > 0:
        hotspot_df = compute_spatial_hotspots(df_map, top_n=5)
        if not hotspot_df.empty:
            st.dataframe(
                hotspot_df,
                column_config={
                    "Country": "Country",
                    "Primary City / Sector": "Cluster Center",
                    "Incidents": st.column_config.NumberColumn("Events", format="%d"),
                    "Fatalities": st.column_config.NumberColumn("Deaths", format="%d"),
                    "Injuries": st.column_config.NumberColumn("Injured", format="%d"),
                    "Lethality (Deaths/Inc)": st.column_config.NumberColumn("Lethality", format="%.2f"),
                    "Dominant Tactic": "Primary Tactic",
                    "Status": st.column_config.TextColumn("Hotspot Status")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("Insufficient geographic cluster density in the active slice.")
    else:
        st.info("No hotspot data to compute for the active filter.")

# Export Snapshot
st.markdown("<br>", unsafe_allow_html=True)
exp_c1, exp_c2 = st.columns(2)
with exp_c1:
    with st.expander("📥 Export Filtered Intelligence Snapshot", expanded=False):
        if total_matches > 0:
            csv_data = df_map.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download Filtered Spatial Slice ({total_matches:,} Records) [CSV]",
                data=csv_data,
                file_name=f"spatial_threat_intel_{active_min_yr}_{active_max_yr}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.caption("No records available to export for the active filter.")

with exp_c2:
    with st.expander("📖 Methodology & Encoding Disclosures", expanded=False):
        st.markdown("""
        * **Dual Visual Encoding:** When set to *All Attack Types*, categorical colors identify attack vectors while alpha opacity and marker size scale dynamically with local incident concentration. When filtering by a single vector, the map switches to a monochromatic intensity scale based on spatial density ($50\\%$), fatalities ($30\\%$), and injuries ($20\\%$).
        * **Mathematical Normalization:** Normalization is evaluated dynamically against the currently filtered dataset. If all points have identical magnitude or zero variance, a fallback median intensity is applied to prevent division-by-zero errors.
        """)

# Auto-Advance Temporal Playback Frame
if st.session_state.get("is_playing", False):
    curr_yr = st.session_state["playback_current_year"]
    if curr_yr < max_yr:
        time.sleep(0.35)
        st.session_state["playback_current_year"] = curr_yr + 1
        st.session_state["map_year_range"] = (min_yr, curr_yr + 1)
        st.rerun()
    else:
        st.session_state["is_playing"] = False
        st.session_state["playback_current_year"] = min_yr
        st.session_state["map_year_range"] = (min_yr, max_yr)
        st.rerun()