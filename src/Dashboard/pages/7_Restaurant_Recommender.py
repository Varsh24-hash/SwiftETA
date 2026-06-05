"""pages/7_Restaurant_Recommender.py – Feature 7: Best Restaurant Recommendation"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import WEATHER_PROFILES, tod_mult, run_p2, compute_route_time, compute_eta, delay_risk

st.set_page_config(page_title="Restaurant Recommender", page_icon="🏆", layout="wide")

st.markdown("# 🏆 Best Restaurant Recommendation")
st.caption("Feature 7 — Now it's optimization, not just prediction  |  Ranks all areas by ETA")
st.divider()

AREAS = {
    "Koramangala":   {"km":4.2,"hw":0.10,"pr":0.40,"se":0.30,"re":0.20,"seg":28,"turns":9},
    "HSR Layout":    {"km":5.8,"hw":0.15,"pr":0.35,"se":0.30,"re":0.20,"seg":35,"turns":11},
    "Indiranagar":   {"km":6.5,"hw":0.08,"pr":0.32,"se":0.38,"re":0.22,"seg":40,"turns":13},
    "Whitefield":    {"km":12.1,"hw":0.25,"pr":0.38,"se":0.22,"re":0.15,"seg":62,"turns":18},
    "Jayanagar":     {"km":7.3,"hw":0.05,"pr":0.28,"se":0.42,"re":0.25,"seg":45,"turns":15},
    "Marathahalli":  {"km":9.4,"hw":0.20,"pr":0.40,"se":0.25,"re":0.15,"seg":52,"turns":16},
    "Electronic City":{"km":14.2,"hw":0.40,"pr":0.35,"se":0.15,"re":0.10,"seg":68,"turns":14},
    "Malleshwaram":  {"km":8.1,"hw":0.05,"pr":0.30,"se":0.40,"re":0.25,"seg":50,"turns":17},
}

CUSTOMER_LOCS = ["Whitefield", "Koramangala", "HSR Layout", "Indiranagar",
                  "Jayanagar", "MG Road (central)"]

with st.sidebar:
    st.markdown("### 🎛️ Parameters")
    cust_loc  = st.selectbox("📍 Customer Location", CUSTOMER_LOCS, index=0)
    weather_k = st.selectbox("🌤️ Weather", list(WEATHER_PROFILES.keys()),
                              format_func=lambda k: WEATHER_PROFILES[k]["label"])
    hour      = st.slider("🕐 Hour", 0, 23, 12)
    seed      = st.number_input("🔢 Seed", 1, 9999, 42)

w_mult = WEATHER_PROFILES[weather_k]["mult"]
t_mult = tod_mult(hour)
p2     = run_p2(seed=int(seed))

# compute ETA from every restaurant area to the customer
results = []
for area, rp in AREAS.items():
    rt  = compute_route_time(rp["km"], rp["hw"], rp["pr"], rp["se"], rp["re"], w_mult, t_mult)
    wd  = round(rt * (w_mult - 1.0), 2)
    eta = compute_eta(rt, p2["cpm_eta"], wd)
    rl, rc, rp_val = delay_risk(eta)
    results.append({"Area": area, "ETA (min)": eta, "Risk": rl,
                    "Late Prob": f"{rp_val:.0%}", "Distance (km)": rp["km"],
                    "_color": rc})

df = pd.DataFrame(results).sort_values("ETA (min)")
best = df.iloc[0]

st.success(f"✅ **Recommended Restaurant Area: {best['Area']}** — ETA: {best['ETA (min)']} min  |  Risk: {best['Risk']}")

col_table, col_chart = st.columns([1, 1], gap="large")

with col_table:
    st.markdown("#### 📋 All Areas Ranked by ETA")
    display_df = df[["Area","ETA (min)","Risk","Late Prob","Distance (km)"]].reset_index(drop=True)
    display_df.index += 1
    st.dataframe(display_df, use_container_width=True)

with col_chart:
    st.markdown("#### 📊 ETA Comparison Chart")
    fig = go.Figure(go.Bar(
        x=df["ETA (min)"], y=df["Area"],
        orientation="h",
        marker_color=df["_color"],
        text=[f"{e} min" for e in df["ETA (min)"]],
        textposition="outside",
    ))
    fig.add_vline(x=30, line_dash="dot", line_color="#ff6b6b",
                  annotation_text="SLA 30 min")
    fig.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
        font_color="#e2e8f0", height=380,
        xaxis_title="ETA (min)", showlegend=False,
        margin=dict(t=10, b=10, l=120, r=60),
    )
    fig.update_xaxes(gridcolor="#2e3148")
    fig.update_yaxes(gridcolor="#2e3148")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown(f"**P2 Stats for this scenario:** CPM ETA = {p2['cpm_eta']} min  |  "
            f"CPM Slack = {p2['cpm_slack']} min  |  "
            f"Batch Size = {p2['batch_size']}")
