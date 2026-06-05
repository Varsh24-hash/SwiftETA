"""
pages/1_ETA_Predictor.py  –  Feature 1: Real-Time ETA Predictor (Main Demo)
User enters: Restaurant Location, Customer Location, Weather, Time of Day, Batch Size
Output: Predicted ETA, Confidence ±, Delay Risk
"""

import streamlit as st
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import (
    WEATHER_PROFILES, tod_mult, run_p2, compute_route_time,
    compute_eta, delay_risk, PROMISED_ETA
)

st.set_page_config(page_title="ETA Predictor", page_icon="🚀", layout="wide")

st.markdown("# Real-Time ETA Predictor")
st.caption("Feature 1 — Main Demo  |  Uses P1 graph features + P2 DAG pipeline")
st.divider()

# ── Bengaluru area presets (lat/lon + avg distance proxy) ────────────────────
AREAS = {
    "Koramangala":  {"km": 4.2, "hw": 0.10, "pr": 0.40, "se": 0.30, "re": 0.20, "seg": 28, "turns": 9},
    "HSR Layout":   {"km": 5.8, "hw": 0.15, "pr": 0.35, "se": 0.30, "re": 0.20, "seg": 35, "turns": 11},
    "Indiranagar":  {"km": 6.5, "hw": 0.08, "pr": 0.32, "se": 0.38, "re": 0.22, "seg": 40, "turns": 13},
    "Whitefield":   {"km": 12.1,"hw": 0.25, "pr": 0.38, "se": 0.22, "re": 0.15, "seg": 62, "turns": 18},
    "Jayanagar":    {"km": 7.3, "hw": 0.05, "pr": 0.28, "se": 0.42, "re": 0.25, "seg": 45, "turns": 15},
    "Marathahalli": {"km": 9.4, "hw": 0.20, "pr": 0.40, "se": 0.25, "re": 0.15, "seg": 52, "turns": 16},
    "Electronic City":{"km":14.2,"hw":0.40, "pr": 0.35, "se": 0.15, "re": 0.10, "seg": 68, "turns": 14},
    "Malleshwaram": {"km": 8.1, "hw": 0.05, "pr": 0.30, "se": 0.40, "re": 0.25, "seg": 50, "turns": 17},
}

# ── sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Order Parameters")
    restaurant = st.selectbox("🍽️ Restaurant Location", list(AREAS.keys()), index=0)
    customer   = st.selectbox("📍 Customer Location",   list(AREAS.keys()), index=2)
    weather_k  = st.selectbox("🌤️ Weather Condition",
                               list(WEATHER_PROFILES.keys()),
                               format_func=lambda k: WEATHER_PROFILES[k]["label"])
    hour       = st.slider("🕐 Time of Day (24h)", 0, 23, 18)
    batch_sz   = st.slider("📦 Batch Size (orders)", 1, 5, 2)
    seed       = st.number_input("🔢 Order Seed (vary for new sample)", 1, 9999, 42)

# ── derive P1 features from area presets ─────────────────────────────────────
r   = AREAS[restaurant]
c   = AREAS[customer]
# combined route: average of both endpoints' road profiles, sum of distances
distance_km  = round((r["km"] + c["km"]) / 2, 2)
road_hw  = (r["hw"] + c["hw"]) / 2
road_pr  = (r["pr"] + c["pr"]) / 2
road_se  = (r["se"] + c["se"]) / 2
road_re  = (r["re"] + c["re"]) / 2
seg_count = (r["seg"] + c["seg"]) // 2
turn_count = (r["turns"] + c["turns"]) // 2

w_mult   = WEATHER_PROFILES[weather_k]["mult"]
t_mult   = tod_mult(hour)

# ── P2 pipeline ───────────────────────────────────────────────────────────────
p2 = run_p2(seed=int(seed), n_window=batch_sz)

# ── ETA components ────────────────────────────────────────────────────────────
route_time    = compute_route_time(distance_km, road_hw, road_pr, road_se, road_re, w_mult, t_mult)
weather_delay = round(route_time * (w_mult - 1.0), 2)
rng           = np.random.default_rng(int(seed))
noise         = float(rng.normal(0, 1))
eta           = compute_eta(route_time, p2["cpm_eta"], weather_delay, noise)
confidence    = round(2.1 + abs(noise) * 0.3, 1)
risk_label, risk_color, risk_prob = delay_risk(eta)

# ── main display ──────────────────────────────────────────────────────────────
col_inputs, col_result = st.columns([1, 1], gap="large")

with col_inputs:
    st.markdown("#### 📋 Order Summary")
    st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Restaurant | **{restaurant}** |
| Customer | **{customer}** |
| Weather | **{WEATHER_PROFILES[weather_k]['label']}** |
| Time | **{hour:02d}:00** {'🔴 Peak' if t_mult > 1.1 else '🟢 Off-peak'} |
| Batch Size | **{batch_sz} orders** |
| Distance (P1) | **{distance_km} km** |
| Segments (P1) | **{seg_count}** |
| Turns (P1) | **{turn_count}** |
""")

with col_result:
    st.markdown("#### 🎯 Prediction Output")
    c1, c2, c3 = st.columns(3)
    c1.metric("⏱️ Predicted ETA", f"{eta} min")
    c2.metric("📊 Confidence", f"±{confidence} min")
    c3.metric("⚠️ Delay Risk",  risk_label,
              delta=f"{risk_prob:.0%} late probability",
              delta_color="inverse")

    st.divider()

    # risk gauge bar
    st.markdown(f"**Delay Risk Meter**")
    fill = int(risk_prob * 20)
    bar  = "█" * fill + "░" * (20 - fill)
    st.markdown(
        f'<p style="font-family:monospace; color:{risk_color}; font-size:18px">'
        f'[{bar}] {risk_prob:.0%}</p>',
        unsafe_allow_html=True,
    )

    late_tag = "🔴 LATE" if eta > PROMISED_ETA else "🟢 ON TIME"
    st.markdown(f"**SLA Status:** {late_tag} (SLA = {PROMISED_ETA} min)")

st.divider()
st.markdown("#### 🧩 ETA Components (from P1 + P2)")
ec1, ec2, ec3, ec4 = st.columns(4)
ec1.metric("🛣️ Route Time (P1)",    f"{route_time} min")
ec2.metric("⚙️ CPM ETA (P2)",       f"{p2['cpm_eta']} min")
ec3.metric("🌧️ Weather Delay",       f"{weather_delay} min")
ec4.metric("📦 CPM Slack (P2)",      f"{p2['cpm_slack']} min")
