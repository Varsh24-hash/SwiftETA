"""
pages/2_Route_Visualizer.py  –  Feature 2: Live Route Visualization
Shows Restaurant → Route → Customer on Bengaluru map with P1 stats.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import WEATHER_PROFILES, tod_mult, run_p2, compute_route_time

st.set_page_config(page_title="Route Visualizer", page_icon="🗺️", layout="wide")

st.markdown("# 🗺️ Live Route Visualization")
st.caption("Feature 2  |  Bengaluru map with P1 graph engine stats")
st.divider()

# ── Bengaluru area coordinates ────────────────────────────────────────────────
COORDS = {
    "Koramangala":   (12.9352, 77.6245),
    "HSR Layout":    (12.9116, 77.6389),
    "Indiranagar":   (12.9784, 77.6408),
    "Whitefield":    (12.9698, 77.7500),
    "Jayanagar":     (12.9250, 77.5938),
    "Marathahalli":  (12.9560, 77.7010),
    "Electronic City":(12.8399, 77.6770),
    "Malleshwaram":  (13.0035, 77.5710),
    "MG Road":       (12.9756, 77.6099),
    "Hebbal":        (13.0350, 77.5970),
}

ROAD_PROFILES = {
    "Koramangala":   {"km":4.2, "hw":0.10,"pr":0.40,"se":0.30,"re":0.20,"seg":28,"turns":9},
    "HSR Layout":    {"km":5.8, "hw":0.15,"pr":0.35,"se":0.30,"re":0.20,"seg":35,"turns":11},
    "Indiranagar":   {"km":6.5, "hw":0.08,"pr":0.32,"se":0.38,"re":0.22,"seg":40,"turns":13},
    "Whitefield":    {"km":12.1,"hw":0.25,"pr":0.38,"se":0.22,"re":0.15,"seg":62,"turns":18},
    "Jayanagar":     {"km":7.3, "hw":0.05,"pr":0.28,"se":0.42,"re":0.25,"seg":45,"turns":15},
    "Marathahalli":  {"km":9.4, "hw":0.20,"pr":0.40,"se":0.25,"re":0.15,"seg":52,"turns":16},
    "Electronic City":{"km":14.2,"hw":0.40,"pr":0.35,"se":0.15,"re":0.10,"seg":68,"turns":14},
    "Malleshwaram":  {"km":8.1, "hw":0.05,"pr":0.30,"se":0.40,"re":0.25,"seg":50,"turns":17},
    "MG Road":       {"km":3.5, "hw":0.08,"pr":0.45,"se":0.30,"re":0.17,"seg":22,"turns":7},
    "Hebbal":        {"km":11.0,"hw":0.30,"pr":0.35,"se":0.20,"re":0.15,"seg":58,"turns":15},
}

with st.sidebar:
    st.markdown("### 🎛️ Route Parameters")
    rest_loc  = st.selectbox("🍽️ Restaurant", list(COORDS.keys()), index=0)
    cust_loc  = st.selectbox("📍 Customer",   list(COORDS.keys()), index=2)
    weather_k = st.selectbox("🌤️ Weather",
                              list(WEATHER_PROFILES.keys()),
                              format_func=lambda k: WEATHER_PROFILES[k]["label"])
    hour      = st.slider("🕐 Hour", 0, 23, 12)

r_coord = COORDS[rest_loc]
c_coord = COORDS[cust_loc]
rp      = ROAD_PROFILES[rest_loc]
cp      = ROAD_PROFILES[cust_loc]

distance_km = round((rp["km"] + cp["km"]) / 2, 1)
seg_count   = (rp["seg"] + cp["seg"]) // 2
turn_count  = (rp["turns"] + cp["turns"]) // 2
road_hw     = round((rp["hw"] + cp["hw"]) / 2, 3)
road_pr     = round((rp["pr"] + cp["pr"]) / 2, 3)
road_se     = round((rp["se"] + cp["se"]) / 2, 3)
road_re     = round((rp["re"] + cp["re"]) / 2, 3)

w_mult   = WEATHER_PROFILES[weather_k]["mult"]
t_mult   = tod_mult(hour)
route_tm = compute_route_time(distance_km, road_hw, road_pr, road_se, road_re, w_mult, t_mult)

# ── midpoints to simulate route line ─────────────────────────────────────────
import numpy as np
mid_lat = (r_coord[0] + c_coord[0]) / 2 + np.random.uniform(-0.005, 0.005)
mid_lon = (r_coord[1] + c_coord[1]) / 2 + np.random.uniform(-0.003, 0.003)
route_lats = [r_coord[0], mid_lat, c_coord[0]]
route_lons = [r_coord[1], mid_lon, c_coord[1]]

# ── map ───────────────────────────────────────────────────────────────────────
fig = go.Figure()

fig.add_trace(go.Scattermapbox(
    lat=route_lats, lon=route_lons,
    mode="lines",
    line=dict(width=4, color="#6c8fff"),
    name="Route",
))
fig.add_trace(go.Scattermapbox(
    lat=[r_coord[0]], lon=[r_coord[1]],
    mode="markers+text",
    marker=dict(size=18, color="#43d9ad"),
    text=[f"🍽️ {rest_loc}"], textposition="top right",
    name="Restaurant",
))
fig.add_trace(go.Scattermapbox(
    lat=[c_coord[0]], lon=[c_coord[1]],
    mode="markers+text",
    marker=dict(size=18, color="#ff6b6b"),
    text=[f"📍 {cust_loc}"], textposition="top right",
    name="Customer",
))

center_lat = (r_coord[0] + c_coord[0]) / 2
center_lon = (r_coord[1] + c_coord[1]) / 2

fig.update_layout(
    mapbox=dict(style="carto-darkmatter", center=dict(lat=center_lat, lon=center_lon), zoom=11),
    margin=dict(l=0, r=0, t=0, b=0),
    height=440,
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0")),
    paper_bgcolor="#0f1117",
)

st.plotly_chart(fig, use_container_width=True)

# ── P1 stats display ─────────────────────────────────────────────────────────
st.markdown("#### 📊 P1 Graph Engine Stats")
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("📏 Distance", f"{distance_km} km")
c2.metric("🔀 Turns",    turn_count)
c3.metric("🔗 Segments", seg_count)
c4.metric("⏱️ Route Time", f"{route_tm} min")
c5.metric("🌧️ Weather Mult", f"{w_mult}×")

st.divider()
st.markdown("#### 🛣️ Road Type Distribution (P1)")
road_df = pd.DataFrame({
    "Road Type": ["Highway", "Primary", "Secondary", "Residential"],
    "Fraction":  [road_hw, road_pr, road_se, road_re],
    "Pct":       [f"{v*100:.0f}%" for v in [road_hw, road_pr, road_se, road_re]],
})

import plotly.express as px
fig2 = px.bar(road_df, x="Road Type", y="Fraction",
              text="Pct",
              color="Road Type",
              color_discrete_sequence=["#6c8fff","#43d9ad","#f5a623","#ff6b6b"])
fig2.update_traces(textposition="outside")
fig2.update_layout(
    paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
    font_color="#e2e8f0", showlegend=False,
    yaxis_tickformat=".0%", height=300,
    margin=dict(t=10, b=10),
)
fig2.update_xaxes(gridcolor="#2e3148")
fig2.update_yaxes(gridcolor="#2e3148")
st.plotly_chart(fig2, use_container_width=True)
