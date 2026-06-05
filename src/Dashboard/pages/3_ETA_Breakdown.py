"""pages/3_ETA_Breakdown.py – Feature 3: ETA Breakdown"""

import streamlit as st
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import WEATHER_PROFILES, tod_mult, run_p2, compute_route_time, compute_eta, PROMISED_ETA

st.set_page_config(page_title="ETA Breakdown", page_icon="⚙️", layout="wide")

st.markdown("# ETA Breakdown")
st.caption("Feature 3 — Instead of 'ETA = X min', see exactly where time goes  |  Uses P2 stage delays")
st.divider()

ROAD_PROFILES = {
    "Koramangala":   {"km":4.2, "hw":0.10,"pr":0.40,"se":0.30,"re":0.20},
    "HSR Layout":    {"km":5.8, "hw":0.15,"pr":0.35,"se":0.30,"re":0.20},
    "Indiranagar":   {"km":6.5, "hw":0.08,"pr":0.32,"se":0.38,"re":0.22},
    "Whitefield":    {"km":12.1,"hw":0.25,"pr":0.38,"se":0.22,"re":0.15},
    "Jayanagar":     {"km":7.3, "hw":0.05,"pr":0.28,"se":0.42,"re":0.25},
    "Marathahalli":  {"km":9.4, "hw":0.20,"pr":0.40,"se":0.25,"re":0.15},
}

with st.sidebar:
    st.markdown("### 🎛️ Parameters")
    area      = st.selectbox("📍 Area", list(ROAD_PROFILES.keys()))
    weather_k = st.selectbox("🌤️ Weather", list(WEATHER_PROFILES.keys()),
                              format_func=lambda k: WEATHER_PROFILES[k]["label"])
    hour      = st.slider("🕐 Hour", 0, 23, 13)
    seed      = st.number_input("🔢 Seed", 1, 9999, 42)

rp       = ROAD_PROFILES[area]
w_mult   = WEATHER_PROFILES[weather_k]["mult"]
t_mult   = tod_mult(hour)
p2       = run_p2(seed=int(seed))

route_time    = compute_route_time(rp["km"], rp["hw"], rp["pr"], rp["se"], rp["re"], w_mult, t_mult)
weather_delay = round(route_time * (w_mult - 1.0), 2)
food_prep     = round(p2["stage_delays"].get("accepted_to_prep", 8.0), 2)
pickup_delay  = round(p2["stage_delays"].get("prep_to_pickup",  3.0), 2)
batching_delay = round((p2["batch_size"] - 1) * 0.8, 2)
total_eta     = round(food_prep + pickup_delay + route_time + weather_delay + batching_delay, 2)

# ── waterfall chart ───────────────────────────────────────────────────────────
labels  = ["Food Preparation", "Pickup Delay", "Travel Time", "Weather Delay", "Batching Delay", "TOTAL ETA"]
values  = [food_prep, pickup_delay, route_time, weather_delay, batching_delay, total_eta]
colors  = ["#43d9ad","#6c8fff","#f5a623","#5599ff","#a78bfa","#ff6b6b"]

fig = go.Figure(go.Bar(
    x=labels, y=values,
    marker_color=colors,
    text=[f"{v} min" for v in values],
    textposition="outside",
))
fig.update_layout(
    paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
    font_color="#e2e8f0", height=380,
    yaxis_title="Minutes", showlegend=False,
    margin=dict(t=20, b=20),
)
fig.update_xaxes(gridcolor="#2e3148")
fig.update_yaxes(gridcolor="#2e3148")
st.plotly_chart(fig, use_container_width=True)

# ── table breakdown ───────────────────────────────────────────────────────────
import pandas as pd
st.markdown("#### 📋 Detailed Breakdown")
breakdown = pd.DataFrame({
    "Stage": ["Food Preparation (P2)", "Pickup Delay (P2)", "Travel Time (P1)",
              "Weather Delay", "Batching Delay (P2)", "── TOTAL ETA ──"],
    "Time (min)": [food_prep, pickup_delay, route_time, weather_delay, batching_delay, total_eta],
    "Source": ["P2 · accepted→prep stage", "P2 · prep→pickup stage",
               "P1 · CH Dijkstra path", "weather_mult × route_time",
               "P2 · batch_size penalty", "Sum of all components"],
})
st.dataframe(breakdown, use_container_width=True, hide_index=True)

late_status = "🔴 LATE" if total_eta > PROMISED_ETA else "🟢 ON TIME"
st.info(f"**{late_status}** — SLA threshold: {PROMISED_ETA} min  |  Total: **{total_eta} min**")
