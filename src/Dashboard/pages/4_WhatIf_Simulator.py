"""pages/4_WhatIf_Simulator.py – Feature 4: What-If Simulator"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import WEATHER_PROFILES, tod_mult, run_p2, compute_route_time, compute_eta, delay_risk

st.set_page_config(page_title="What-If Simulator", page_icon="🌧️", layout="wide")

st.markdown("# What-If Simulator")
st.caption("Feature 4 — Tweak conditions and see ETA change in real time ")
st.divider()

BASE_DIST = 7.0
BASE_HW, BASE_PR, BASE_SE, BASE_RE = 0.15, 0.38, 0.30, 0.17

with st.sidebar:
    st.markdown("### 🎛️ Simulation Controls")
    weather_k    = st.selectbox("🌤️ Weather", list(WEATHER_PROFILES.keys()),
                                 format_func=lambda k: WEATHER_PROFILES[k]["label"])
    hour         = st.slider("🕐 Hour of Day", 0, 23, 18)
    batch_size   = st.slider("📦 Batch Size", 1, 6, 2)
    traffic_pct  = st.slider("🚗 Traffic Level (%)", 0, 100, 50,
                              help="0 = free flow, 100 = gridlock")
    distance_km  = st.slider("📏 Route Distance (km)", 1.0, 20.0, BASE_DIST, 0.5)
    seed         = st.number_input("🔢 Seed", 1, 9999, 42)

# ── compute for all 4 weather scenarios ───────────────────────────────────────
traffic_mult = 1.0 + (traffic_pct / 100) * 0.5   # 1.0 → 1.5×
t_mult       = tod_mult(hour) * traffic_mult
p2           = run_p2(seed=int(seed), n_window=batch_size)

scenarios = {}
for wk, wp in WEATHER_PROFILES.items():
    wm    = wp["mult"]
    rt    = compute_route_time(distance_km, BASE_HW, BASE_PR, BASE_SE, BASE_RE, wm, t_mult)
    wd    = round(rt * (wm - 1.0), 2)
    bd    = round((batch_size - 1) * 0.8, 2)
    eta   = compute_eta(rt, p2["cpm_eta"], wd + bd)
    rl, rc, rp = delay_risk(eta)
    scenarios[wk] = {"eta": eta, "risk": rl, "color": rc, "prob": rp,
                     "route": rt, "weather_d": wd, "batch_d": bd}

# ── highlight selected scenario ───────────────────────────────────────────────
sel = scenarios[weather_k]
c1, c2, c3 = st.columns(3)
c1.metric("⏱️ Predicted ETA",   f"{sel['eta']} min")
c2.metric("⚠️ Delay Risk",      sel["risk"])
c3.metric("🎲 Late Probability", f"{sel['prob']:.0%}")

st.divider()

# ── comparison bar chart across all weather scenarios ─────────────────────────
st.markdown("#### 📊 ETA Across Weather Scenarios")
labels = [WEATHER_PROFILES[k]["label"] for k in WEATHER_PROFILES]
etas   = [scenarios[k]["eta"] for k in WEATHER_PROFILES]
cols   = [scenarios[k]["color"] for k in WEATHER_PROFILES]

fig = go.Figure(go.Bar(
    x=labels, y=etas,
    marker_color=cols,
    text=[f"{e} min" for e in etas],
    textposition="outside",
))
fig.add_hline(y=30, line_dash="dot", line_color="#ff6b6b",
              annotation_text="SLA 30 min", annotation_position="right")
fig.update_layout(
    paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
    font_color="#e2e8f0", height=340, showlegend=False,
    yaxis_title="ETA (min)", margin=dict(t=10, b=10),
)
fig.update_xaxes(gridcolor="#2e3148")
fig.update_yaxes(gridcolor="#2e3148")
st.plotly_chart(fig, use_container_width=True)

# ── hour sweep ────────────────────────────────────────────────────────────────
st.markdown("#### 🕐 ETA vs Hour of Day (current weather + distance)")
hours = list(range(24))
etas_h = []
for h in hours:
    tm  = tod_mult(h) * traffic_mult
    rt  = compute_route_time(distance_km, BASE_HW, BASE_PR, BASE_SE, BASE_RE,
                              WEATHER_PROFILES[weather_k]["mult"], tm)
    wd  = rt * (WEATHER_PROFILES[weather_k]["mult"] - 1.0)
    etas_h.append(compute_eta(rt, p2["cpm_eta"], wd))

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=hours, y=etas_h, mode="lines+markers",
                           line=dict(color="#6c8fff", width=2),
                           marker=dict(size=5)))
fig2.add_hline(y=30, line_dash="dot", line_color="#ff6b6b")
fig2.add_vrect(x0=8, x1=10, fillcolor="#ff6b6b", opacity=0.08, line_width=0,
               annotation_text="AM Peak", annotation_position="top left")
fig2.add_vrect(x0=17, x1=20, fillcolor="#ff6b6b", opacity=0.08, line_width=0,
               annotation_text="PM Peak", annotation_position="top left")
fig2.update_layout(
    paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
    font_color="#e2e8f0", height=300,
    xaxis_title="Hour of Day", yaxis_title="ETA (min)",
    margin=dict(t=10, b=10),
)
fig2.update_xaxes(gridcolor="#2e3148", dtick=2)
fig2.update_yaxes(gridcolor="#2e3148")
st.plotly_chart(fig2, use_container_width=True)
