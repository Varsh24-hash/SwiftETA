"""pages/5_Delay_Risk_Meter.py – Feature 5: Delay Risk Meter"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import WEATHER_PROFILES, tod_mult, run_p2, compute_route_time, compute_eta, delay_risk, PROMISED_ETA

st.set_page_config(page_title="Delay Risk Meter", page_icon="🔥", layout="wide")

st.markdown("# Delay Risk Meter")
st.caption("Feature 5 — Like a fuel gauge for delivery risk  |  This feels like a production product")
st.divider()

ROAD_PROFILES = {
    "Koramangala":  {"km":4.2,"hw":0.10,"pr":0.40,"se":0.30,"re":0.20},
    "HSR Layout":   {"km":5.8,"hw":0.15,"pr":0.35,"se":0.30,"re":0.20},
    "Indiranagar":  {"km":6.5,"hw":0.08,"pr":0.32,"se":0.38,"re":0.22},
    "Whitefield":   {"km":12.1,"hw":0.25,"pr":0.38,"se":0.22,"re":0.15},
    "Marathahalli": {"km":9.4,"hw":0.20,"pr":0.40,"se":0.25,"re":0.15},
    "Electronic City":{"km":14.2,"hw":0.40,"pr":0.35,"se":0.15,"re":0.10},
}

with st.sidebar:
    st.markdown("### 🎛️ Order Parameters")
    area      = st.selectbox("📍 Area", list(ROAD_PROFILES.keys()), index=3)
    weather_k = st.selectbox("🌤️ Weather", list(WEATHER_PROFILES.keys()), index=2,
                              format_func=lambda k: WEATHER_PROFILES[k]["label"])
    hour      = st.slider("🕐 Hour", 0, 23, 18)
    batch_sz  = st.slider("📦 Batch Size", 1, 5, 3)
    seed      = st.number_input("🔢 Seed", 1, 9999, 42)

rp        = ROAD_PROFILES[area]
w_mult    = WEATHER_PROFILES[weather_k]["mult"]
t_mult    = tod_mult(hour)
p2        = run_p2(seed=int(seed), n_window=batch_sz)
route_tm  = compute_route_time(rp["km"], rp["hw"], rp["pr"], rp["se"], rp["re"], w_mult, t_mult)
w_delay   = round(route_tm * (w_mult - 1.0), 2)
eta       = compute_eta(route_tm, p2["cpm_eta"], w_delay)
rl, rc, rp_val = delay_risk(eta)

# ── gauge ─────────────────────────────────────────────────────────────────────
fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=round(rp_val * 100, 1),
    title={"text": "Delay Probability (%)", "font": {"color": "#e2e8f0", "size": 18}},
    delta={"reference": 35, "increasing": {"color": "#ff6b6b"},
           "decreasing": {"color": "#43d9ad"}},
    gauge={
        "axis": {"range": [0, 100], "tickcolor": "#8892a4",
                 "tickfont": {"color": "#8892a4"}},
        "bar":  {"color": rc, "thickness": 0.25},
        "bgcolor": "#1a1d27",
        "bordercolor": "#2e3148",
        "steps": [
            {"range": [0, 35],   "color": "#0d2e1f"},
            {"range": [35, 65],  "color": "#2e2410"},
            {"range": [65, 100], "color": "#2e1010"},
        ],
        "threshold": {
            "line": {"color": "#ff6b6b", "width": 3},
            "thickness": 0.8,
            "value": 65,
        },
    },
    number={"suffix": "%", "font": {"color": rc, "size": 40}},
))
fig.update_layout(paper_bgcolor="#0f1117", font_color="#e2e8f0", height=380,
                  margin=dict(t=30, b=10, l=30, r=30))
st.plotly_chart(fig, use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("⏱️ ETA",        f"{eta} min")
c2.metric("⚠️ Risk Level", rl)
c3.metric("📊 Late Prob",  f"{rp_val:.0%}")
c4.metric("📦 CPM Slack",  f"{p2['cpm_slack']} min")

st.divider()
st.markdown("#### 🎯 Risk Thresholds")
st.markdown("""
| Risk Level | ETA Range | Late Probability | Action |
|-----------|-----------|-----------------|--------|
| 🟢 **LOW** | < 25 min | < 35% | Normal dispatch |
| 🟡 **MEDIUM** | 25–33 min | 35–65% | Alert restaurant |
| 🔴 **HIGH** | > 33 min | > 65% | Proactive customer notification |
""")
