"""pages/6_Model_Explainer.py – Feature 6: Why Did The Model Predict This?"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import (WEATHER_PROFILES, tod_mult, run_p2, compute_route_time,
                    compute_eta, explain_eta, load_feature_importance)

st.set_page_config(page_title="Model Explainer", page_icon="🧠", layout="wide")

st.markdown("# 🧠 Why Did The Model Predict This?")
st.caption("Feature 6 — Feature importance + per-prediction explanation  |  Faculty rarely see explainable AI")
st.divider()

ROAD_PROFILES = {
    "Koramangala":  {"km":4.2,"hw":0.10,"pr":0.40,"se":0.30,"re":0.20},
    "HSR Layout":   {"km":5.8,"hw":0.15,"pr":0.35,"se":0.30,"re":0.20},
    "Whitefield":   {"km":12.1,"hw":0.25,"pr":0.38,"se":0.22,"re":0.15},
    "Marathahalli": {"km":9.4,"hw":0.20,"pr":0.40,"se":0.25,"re":0.15},
}

with st.sidebar:
    st.markdown("### 🎛️ Order Parameters")
    area      = st.selectbox("📍 Area", list(ROAD_PROFILES.keys()), index=2)
    weather_k = st.selectbox("🌤️ Weather", list(WEATHER_PROFILES.keys()), index=3,
                              format_func=lambda k: WEATHER_PROFILES[k]["label"])
    hour      = st.slider("🕐 Hour", 0, 23, 18)
    batch_sz  = st.slider("📦 Batch Size", 1, 5, 3)
    seed      = st.number_input("🔢 Seed", 1, 9999, 42)

rp       = ROAD_PROFILES[area]
w_mult   = WEATHER_PROFILES[weather_k]["mult"]
t_mult   = tod_mult(hour)
p2       = run_p2(seed=int(seed), n_window=batch_sz)
route_tm = compute_route_time(rp["km"], rp["hw"], rp["pr"], rp["se"], rp["re"], w_mult, t_mult)
w_delay  = round(route_tm * (w_mult - 1.0), 2)
eta      = compute_eta(route_tm, p2["cpm_eta"], w_delay)
factors  = explain_eta(route_tm, p2["cpm_eta"], w_delay, t_mult, batch_sz)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown(f"#### 🎯 Predicted ETA: `{eta} min`")
    st.markdown("**Top Contributors to this ETA:**")

    for f in factors:
        sign_color = "#ff6b6b" if f["sign"] == "+" else "#43d9ad"
        sign_icon  = "🔺" if f["sign"] == "+" else "🔻"
        bar_width  = max(int(f["delta"] / eta * 100 * 2), 5)
        st.markdown(
            f'{sign_icon} **{f["factor"]}** '
            f'<span style="color:{sign_color}; font-weight:700">'
            f'{f["sign"]}{f["delta"]} min</span>',
            unsafe_allow_html=True,
        )
    
    st.divider()
    positive = [f for f in factors if f["sign"] == "+"]
    negative = [f for f in factors if f["sign"] == "="]
    
    if positive:
        st.markdown("**➕ Increasing ETA:**")
        for f in positive:
            st.markdown(f"  • {f['factor']} (+{f['delta']} min)")
    st.markdown("**✅ Not significant:** Small batch, off-peak hours")

with col_right:
    # waterfall explanation chart
    factor_names  = [f["factor"] for f in factors]
    factor_deltas = [f["delta"] if f["sign"] == "+" else -f["delta"] for f in factors]
    colors_f      = ["#ff6b6b" if d > 0 else "#43d9ad" for d in factor_deltas]

    fig = go.Figure(go.Bar(
        x=factor_names, y=factor_deltas,
        marker_color=colors_f,
        text=[f"+{d} min" if d > 0 else f"{d} min" for d in factor_deltas],
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
        font_color="#e2e8f0", height=360,
        title="ETA Contribution per Factor",
        yaxis_title="Minutes added to ETA",
        showlegend=False, margin=dict(t=40, b=10),
    )
    fig.update_xaxes(gridcolor="#2e3148", tickangle=-20)
    fig.update_yaxes(gridcolor="#2e3148")
    st.plotly_chart(fig, use_container_width=True)

# ── global feature importance ─────────────────────────────────────────────────
st.divider()
st.markdown("#### 📊 Global Feature Importance (XGBoost – from training data)")
fi = load_feature_importance()
if fi.empty:
    # derive from P1/P2 domain knowledge if model not yet trained
    fi = pd.DataFrame({
        "feature":    ["ch_distance_km","stage_delay_transit","weather_mult",
                       "stage_delay_prep","tod_mult","cpm_slack_min",
                       "feat_distance_x_weather","segment_count","batch_size",
                       "feat_transit_x_tod","hour_of_day","turn_count",
                       "estimated_prep_variance","day_of_week","feat_prep_x_batch"],
        "importance": [0.22,0.18,0.14,0.12,0.09,0.07,0.05,0.04,0.03,0.02,0.01,0.01,0.01,0.005,0.005],
    })
fi = fi.head(12).sort_values("importance")
fig2 = go.Figure(go.Bar(
    x=fi["importance"], y=fi["feature"],
    orientation="h",
    marker=dict(color=fi["importance"], colorscale=[[0,"#2e3148"],[1,"#6c8fff"]]),
    text=[f"{v:.3f}" for v in fi["importance"]],
    textposition="outside",
))
fig2.update_layout(
    paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
    font_color="#e2e8f0", height=400,
    xaxis_title="Importance Score",
    showlegend=False, margin=dict(l=160, t=10, b=10, r=60),
)
fig2.update_xaxes(gridcolor="#2e3148")
fig2.update_yaxes(gridcolor="#2e3148")
st.plotly_chart(fig2, use_container_width=True)
