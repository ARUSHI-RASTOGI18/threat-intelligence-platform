"""
Module 2: Zero-API Offline Global Threat Geospatial Map
Location: ./pages/1_Global_Threat_Map.py
"""

import streamlit as st
import plotly.express as px
import pandas as pd
from src.data_loader import load_analytical_data

st.set_page_config(page_title="Global Threat Map | GTI-ARP", page_icon="🗺️", layout="wide")

df = load_analytical_data()

st.title("Geospatial Threat Distribution Map")
st.markdown("Interactive spatial analytics with Dual Render Modes. Fully offline-compatible.")

f1, f2, f3, f4 = st.columns(4)
with f1:
    min_yr, max_yr = int(df["year"].min()), int(df["year"].max())
    map_years = st.slider("Year Horizon", min_yr, max_yr, (min_yr, max_yr))
with f2:
    regions = ["All Regions"] + sorted(df["region"].unique().tolist())
    selected_region = st.selectbox("Region Filter", regions)
with f3:
    attacks = ["All Attack Types"] + sorted(df["attack_type"].unique().tolist())
    selected_attack = st.selectbox("Attack Type Filter", attacks)
with f4:
    map_mode = st.radio("Map Projection", ["Point Scatter (Geo)", "Lethality Density Heatmap"], horizontal=True)

df_map = df[(df["year"] >= map_years[0]) & (df["year"] <= map_years[1])]
if selected_region != "All Regions":
    df_map = df_map[df_map["region"] == selected_region]
if selected_attack != "All Attack Types":
    df_map = df_map[df_map["attack_type"] == selected_attack]

df_map_geo = df_map.dropna(subset=["latitude", "longitude"]).copy()
total_matches = len(df_map_geo)

MAX_POINTS = 6000
is_sampled = False
if total_matches > MAX_POINTS:
    df_plot = df_map_geo.sample(n=MAX_POINTS, random_state=42)
    is_sampled = True
else:
    df_plot = df_map_geo

# Transparency Telemetry
t1, t2, t3 = st.columns(3)
t1.metric("Matching Geocoded Records", f"{total_matches:,}")
t2.metric("Rendered Sample Points", f"{len(df_plot):,}")
t3.metric("Sampling Status", "Active (Uniform Random Sample)" if is_sampled else "Full Coverage (100%)")

if df_plot.empty:
    st.error("No geocoded incidents match the selected filter parameters.")
else:
    if map_mode == "Point Scatter (Geo)":
        fig_map = px.scatter_geo(
            df_plot,
            lat="latitude",
            lon="longitude",
            hover_name="city",
            hover_data={
                "country": True,
                "year": True,
                "attack_type": True,
                "target_type": True,
                "weapon_type": True,
                "fatalities": True,
                "injured": True,
                "success": True,
                "latitude": False,
                "longitude": False
            },
            color="attack_type",
            size="fatalities",
            size_max=14,
            projection="natural earth",
            template="plotly_dark",
            title=f"Spatial Incident Scatter ({len(df_plot):,} events rendered)"
        )
        fig_map.update_geos(
            showcountries=True, countrycolor="#30363D",
            showocean=True, oceancolor="#0E1117",
            showland=True, landcolor="#161B22",
            bgcolor="#0E1117"
        )
    else:
        fig_map = px.density_mapbox(
            df_plot,
            lat="latitude",
            lon="longitude",
            z="fatalities",
            radius=10,
            zoom=1.1,
            mapbox_style="open-street-map",
            template="plotly_dark",
            title=f"Lethality Density Heatmap"
        )

    fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=620)
    st.plotly_chart(fig_map, use_container_width=True)