"""pages/8_Peak_Analytics.py – Feature 8: Peak-Hour Analytics"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import WEATHER_PROFILES, tod_mult, run_p2, compute_route_time, compute_eta, load_all_orders

st.set_page_config(page_title="Peak-Hour Analytics", page_icon="📈", layout="wide")

st.markdown("# 📈 Peak-Hour Analytics")
st.caption("Feature 8 — Hour vs Average ETA  |  Faculty immediately understand: 8-10 AM spike, 5-8 PM spike")
st.divider()

BASE_KM = 7.0
BASE_HW, BASE_PR, BASE_SE, BASE_RE = 0.15, 0.38, 0.30, 0.17

with st.sidebar:
    st.markdown("### 🎛️ Parameters")
    weather_k = st.selectbox("🌤️ Weather", list(WEATHER_PROFILES.keys()),
                              format_func=lambda k: WEATHER_PROFILES[k]["label"])
    day_type  = st.radio("📅 Day Type", ["Weekday", "Weekend"])
    seed      = st.number_input("🔢 Seed", 1, 9999, 42)

p2     = run_p2(seed=int(seed))
w_mult = WEATHER_PROFILES[weather_k]["mult"]

# weekend has slightly reduced congestion
weekend_factor = 0.80 if day_type == "Weekend" else 1.0

hours = list(range(24))
etas, t_mults = [], []
for h in hours:
    tm  = tod_mult(h) * weekend_factor
    rt  = compute_route_time(BASE_KM, BASE_HW, BASE_PR, BASE_SE, BASE_RE, w_mult, tm)
    wd  = rt * (w_mult - 1.0)
    eta = compute_eta(rt, p2["cpm_eta"], wd)
    etas.append(eta)
    t_mults.append(round(tm, 2))

peak_hour   = hours[etas.index(max(etas))]
trough_hour = hours[etas.index(min(etas))]

c1, c2, c3 = st.columns(3)
c1.metric("🔴 Worst Hour",  f"{peak_hour:02d}:00",   delta=f"ETA: {max(etas)} min")
c2.metric("🟢 Best Hour",   f"{trough_hour:02d}:00",  delta=f"ETA: {min(etas)} min", delta_color="off")
c3.metric("📊 Peak Penalty", f"+{round(max(etas)-min(etas),1)} min")

# ── line chart ────────────────────────────────────────────────────────────────
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hours, y=etas,
    mode="lines+markers",
    line=dict(color="#6c8fff", width=2.5),
    marker=dict(size=6, color=["#ff6b6b" if (8<=h<=10 or 17<=h<=20) else "#6c8fff" for h in hours]),
    name="Avg ETA",
    fill="tozeroy", fillcolor="rgba(108,143,255,0.08)",
))
fig.add_hline(y=30, line_dash="dot", line_color="#ff6b6b",
              annotation_text="SLA 30 min", annotation_position="bottom right")
fig.add_vrect(x0=8, x1=10, fillcolor="#ff6b6b", opacity=0.10, line_width=0,
              annotation_text="AM Peak 8–10", annotation_position="top left",
              annotation_font_color="#ff6b6b")
fig.add_vrect(x0=17, x1=20, fillcolor="#ff6b6b", opacity=0.10, line_width=0,
              annotation_text="PM Peak 5–8", annotation_position="top left",
              annotation_font_color="#ff6b6b")
fig.add_vrect(x0=0, x1=6, fillcolor="#43d9ad", opacity=0.05, line_width=0,
              annotation_text="Night", annotation_position="top left",
              annotation_font_color="#43d9ad")
fig.update_layout(
    paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
    font_color="#e2e8f0", height=400,
    xaxis_title="Hour of Day", yaxis_title="Average ETA (min)",
    xaxis=dict(tickmode="linear", dtick=2, gridcolor="#2e3148"),
    yaxis=dict(gridcolor="#2e3148"),
    margin=dict(t=20, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# ── hourly table ─────────────────────────────────────────────────────────────
import pandas as pd
st.markdown("#### 📋 Hourly ETA Table")
hour_df = pd.DataFrame({
    "Hour":        [f"{h:02d}:00" for h in hours],
    "ETA (min)":   etas,
    "ToD Mult":    t_mults,
    "Period":      ["🔴 AM Peak" if 8<=h<=10 else "🔴 PM Peak" if 17<=h<=20
                    else "⚪ Off-Peak" for h in hours],
})
st.dataframe(hour_df, use_container_width=True, hide_index=True, height=250)

# pull from real data if available
all_df = load_all_orders()
if not all_df.empty:
    st.divider()
    st.markdown("#### 📊 Real Data: Avg ETA by Hour (from collected orders)")
    real_hourly = all_df.groupby("hour_of_day")["ground_truth_eta"].mean().reset_index()
    fig2 = go.Figure(go.Bar(
        x=real_hourly["hour_of_day"], y=real_hourly["ground_truth_eta"].round(1),
        marker_color=["#ff6b6b" if (8<=h<=10 or 17<=h<=20) else "#6c8fff"
                      for h in real_hourly["hour_of_day"]],
        text=real_hourly["ground_truth_eta"].round(1),
        textposition="outside",
    ))
    fig2.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                       font_color="#e2e8f0", height=320,
                       xaxis_title="Hour of Day", yaxis_title="Avg ETA (min)",
                       margin=dict(t=10, b=10))
    fig2.update_xaxes(gridcolor="#2e3148")
    fig2.update_yaxes(gridcolor="#2e3148")
    st.plotly_chart(fig2, use_container_width=True)
