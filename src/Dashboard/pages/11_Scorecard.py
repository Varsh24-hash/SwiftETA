"""pages/11_Scorecard.py – Feature 13: Delivery Scorecard"""

import streamlit as st
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import WEATHER_PROFILES, tod_mult, run_p2, compute_route_time, compute_eta, scorecard

st.set_page_config(page_title="Delivery Scorecard", page_icon="🥇", layout="wide")

st.markdown("# Delivery Scorecard")
st.caption("Feature 13 — Route / Traffic / Weather / Ops scores")
st.divider()

ROAD_PROFILES = {
    "Koramangala":  {"km":4.2,"hw":0.10,"pr":0.40,"se":0.30,"re":0.20},
    "HSR Layout":   {"km":5.8,"hw":0.15,"pr":0.35,"se":0.30,"re":0.20},
    "Indiranagar":  {"km":6.5,"hw":0.08,"pr":0.32,"se":0.38,"re":0.22},
    "Whitefield":   {"km":12.1,"hw":0.25,"pr":0.38,"se":0.22,"re":0.15},
    "Marathahalli": {"km":9.4,"hw":0.20,"pr":0.40,"se":0.25,"re":0.15},
}

with st.sidebar:
    st.markdown("### 🎛️ Order Parameters")
    area      = st.selectbox("📍 Area", list(ROAD_PROFILES.keys()), index=0)
    weather_k = st.selectbox("🌤️ Weather", list(WEATHER_PROFILES.keys()),
                              format_func=lambda k: WEATHER_PROFILES[k]["label"])
    hour      = st.slider("🕐 Hour", 0, 23, 14)
    seed      = st.number_input("🔢 Seed", 1, 9999, 42)

rp       = ROAD_PROFILES[area]
w_mult   = WEATHER_PROFILES[weather_k]["mult"]
t_mult   = tod_mult(hour)
p2       = run_p2(seed=int(seed))
rt       = compute_route_time(rp["km"], rp["hw"], rp["pr"], rp["se"], rp["re"], w_mult, t_mult)
wd       = round(rt * (w_mult - 1.0), 2)
eta      = compute_eta(rt, p2["cpm_eta"], wd)
scores   = scorecard(rp["km"], w_mult, t_mult, p2["batch_size"], p2["cpm_slack"])

# ── overall score gauge ───────────────────────────────────────────────────────
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=scores["Overall"],
    title={"text": "Overall Delivery Score", "font": {"color":"#e2e8f0","size":18}},
    gauge={
        "axis": {"range":[0,100], "tickcolor":"#8892a4", "tickfont":{"color":"#8892a4"}},
        "bar":  {"color": "#43d9ad" if scores["Overall"]>=70 else "#f5a623" if scores["Overall"]>=50 else "#ff6b6b",
                 "thickness":0.25},
        "bgcolor": "#1a1d27", "bordercolor":"#2e3148",
        "steps": [
            {"range":[0,50],  "color":"#2e1010"},
            {"range":[50,70], "color":"#2e2410"},
            {"range":[70,100],"color":"#0d2e1f"},
        ],
        "threshold":{"line":{"color":"#43d9ad","width":3},"thickness":0.8,"value":70},
    },
    number={"suffix":"/100","font":{"color":"#43d9ad","size":48}},
))
fig_gauge.update_layout(paper_bgcolor="#0f1117", font_color="#e2e8f0",
                         height=340, margin=dict(t=30,b=10,l=30,r=30))
st.plotly_chart(fig_gauge, use_container_width=True)

# ── individual scores ─────────────────────────────────────────────────────────
st.markdown("#### 📊 Score Breakdown")
c1,c2,c3,c4 = st.columns(4)
score_items = [
    (c1, "🛣️ Route Efficiency",  scores["Route Efficiency"],  "Short, highway-heavy route"),
    (c2, "🚗 Traffic Impact",     scores["Traffic Impact"],    "Lower = worse traffic"),
    (c3, "🌧️ Weather Impact",     scores["Weather Impact"],    "Lower = worse weather"),
    (c4, "⚙️ Operational",        scores["Operational"],       "CPM slack + batch efficiency"),
]
for col, label, val, desc in score_items:
    color = "#43d9ad" if val>=70 else "#f5a623" if val>=50 else "#ff6b6b"
    col.markdown(
        f'<div style="background:#1a1d27;border:1px solid #2e3148;border-radius:10px;padding:16px;text-align:center">'
        f'<div style="color:#8892a4;font-size:13px">{label}</div>'
        f'<div style="color:{color};font-size:32px;font-weight:800">{val}</div>'
        f'<div style="color:#8892a4;font-size:11px">{desc}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── radar chart ───────────────────────────────────────────────────────────────
st.divider()
cats = ["Route Efficiency","Traffic Impact","Weather Impact","Operational"]
vals = [scores[c] for c in cats] + [scores[cats[0]]]
cats_plot = cats + [cats[0]]

fig_radar = go.Figure(go.Scatterpolar(
    r=vals, theta=cats_plot,
    fill="toself", fillcolor="rgba(108,143,255,0.15)",
    line=dict(color="#6c8fff", width=2),
    marker=dict(color="#6c8fff", size=8),
))
fig_radar.update_layout(
    polar=dict(
        bgcolor="#1a1d27",
        radialaxis=dict(visible=True, range=[0,100], tickcolor="#8892a4",
                        gridcolor="#2e3148", linecolor="#2e3148"),
        angularaxis=dict(tickcolor="#e2e8f0", gridcolor="#2e3148", linecolor="#2e3148"),
    ),
    paper_bgcolor="#0f1117", font_color="#e2e8f0",
    height=380, margin=dict(t=20,b=20,l=60,r=60),
    title=f"Scorecard Radar — {area}  {WEATHER_PROFILES[weather_k]['label']}  {hour:02d}:00",
)
st.plotly_chart(fig_radar, use_container_width=True)
st.info(f"**ETA:** {eta} min  |  **CPM Slack:** {p2['cpm_slack']} min  |  **Batch Size:** {p2['batch_size']}")
