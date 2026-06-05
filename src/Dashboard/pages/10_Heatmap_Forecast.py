"""pages/10_Heatmap_Forecast.py – Feature 11: Delivery Heatmap + Feature 12: ETA Forecast"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import WEATHER_PROFILES, tod_mult, run_p2, compute_route_time, compute_eta, load_all_orders, PROMISED_ETA

st.set_page_config(page_title="Heatmap & Forecast", page_icon="🎯", layout="wide")

st.markdown("# 🎯 Delivery Heatmap + ETA Forecast")
st.caption("Feature 11 + 12 — Green=Fast / Red=Slow on Bengaluru map · Predict ETA distribution for future orders")
st.divider()

BENGALURU_AREAS = {
    "Koramangala":   (12.9352, 77.6245, 4.2),
    "HSR Layout":    (12.9116, 77.6389, 5.8),
    "Indiranagar":   (12.9784, 77.6408, 6.5),
    "Whitefield":    (12.9698, 77.7500, 12.1),
    "Jayanagar":     (12.9250, 77.5938, 7.3),
    "Marathahalli":  (12.9560, 77.7010, 9.4),
    "Electronic City":(12.8399,77.6770, 14.2),
    "Malleshwaram":  (13.0035, 77.5710, 8.1),
    "MG Road":       (12.9756, 77.6099, 3.5),
    "Hebbal":        (13.0350, 77.5970, 11.0),
    "Yelahanka":     (13.1007, 77.5963, 13.5),
    "JP Nagar":      (12.9081, 77.5916, 6.8),
}
ROAD_HW_PR_SE_RE = {  # highway, primary, secondary, residential
    "Koramangala":   (0.10,0.40,0.30,0.20),
    "HSR Layout":    (0.15,0.35,0.30,0.20),
    "Indiranagar":   (0.08,0.32,0.38,0.22),
    "Whitefield":    (0.25,0.38,0.22,0.15),
    "Jayanagar":     (0.05,0.28,0.42,0.25),
    "Marathahalli":  (0.20,0.40,0.25,0.15),
    "Electronic City":(0.40,0.35,0.15,0.10),
    "Malleshwaram":  (0.05,0.30,0.40,0.25),
    "MG Road":       (0.08,0.45,0.30,0.17),
    "Hebbal":        (0.30,0.35,0.20,0.15),
    "Yelahanka":     (0.28,0.32,0.25,0.15),
    "JP Nagar":      (0.05,0.30,0.40,0.25),
}

tab1, tab2 = st.tabs(["🗺️ Delivery Heatmap", "🔮 ETA Forecast"])

with tab1:
    col_s1, col_s2 = st.columns([1,3])
    with col_s1:
        weather_k = st.selectbox("🌤️ Weather", list(WEATHER_PROFILES.keys()),
                                  format_func=lambda k: WEATHER_PROFILES[k]["label"], key="hw")
        hour      = st.slider("🕐 Hour", 0, 23, 14, key="hh")
        seed      = st.number_input("🔢 Seed", 1, 9999, 42, key="hs")

    w_mult = WEATHER_PROFILES[weather_k]["mult"]
    t_mult = tod_mult(hour)
    p2     = run_p2(seed=int(seed))

    map_rows = []
    for area, (lat, lon, km) in BENGALURU_AREAS.items():
        hw,pr,se,re = ROAD_HW_PR_SE_RE[area]
        rt  = compute_route_time(km, hw, pr, se, re, w_mult, t_mult)
        wd  = rt * (w_mult - 1.0)
        eta = compute_eta(rt, p2["cpm_eta"], wd)
        map_rows.append({"area": area, "lat": lat, "lon": lon, "eta": round(eta,1),
                          "status": "Slow (Late)" if eta > 30 else "Fast (On Time)"})
    map_df = pd.DataFrame(map_rows)

    fig = px.scatter_mapbox(
        map_df, lat="lat", lon="lon", color="status",
        size="eta", size_max=30,
        color_discrete_map={"Fast (On Time)": "#43d9ad", "Slow (Late)": "#ff6b6b"},
        hover_data={"eta": True, "area": True, "lat": False, "lon": False},
        text="area",
        mapbox_style="carto-darkmatter",
        zoom=10.5, center={"lat": 12.9716, "lon": 77.6412},
        title=f"Delivery Heatmap — {WEATHER_PROFILES[weather_k]['label']}  {hour:02d}:00",
    )
    fig.update_layout(paper_bgcolor="#0f1117", font_color="#e2e8f0",
                       height=500, margin=dict(l=0,r=0,t=40,b=0))
    with col_s2:
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 Area ETA Summary")
    st.dataframe(map_df[["area","eta","status"]].sort_values("eta")
                 .rename(columns={"area":"Area","eta":"ETA (min)","status":"Status"}),
                 use_container_width=True, hide_index=True)

with tab2:
    st.markdown("### 🔮 ETA Forecast for Future Orders")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        future_day  = st.selectbox("📅 Day", ["Today","Tomorrow","Day After"], index=1)
    with col_f2:
        future_hour = st.selectbox("🕐 Time Slot", [f"{h:02d}:00" for h in range(0,24,1)], index=19)
    with col_f3:
        future_wx   = st.selectbox("🌤️ Expected Weather", list(WEATHER_PROFILES.keys()), index=3,
                                    format_func=lambda k: WEATHER_PROFILES[k]["label"])

    fh     = int(future_hour.split(":")[0])
    fw     = WEATHER_PROFILES[future_wx]["mult"]
    ft     = tod_mult(fh)

    # Simulate ETA distribution across 200 orders at those conditions
    etas_future = []
    for s in range(200):
        p2f = run_p2(seed=s+1000)
        km  = np.random.uniform(3, 14)
        hw,pr,se,re = 0.15,0.38,0.30,0.17
        rt  = compute_route_time(km, hw, pr, se, re, fw, ft)
        wd  = rt * (fw - 1.0)
        eta = compute_eta(rt, p2f["cpm_eta"], wd, float(np.random.normal(0,1)))
        etas_future.append(eta)
    etas_arr = np.array(etas_future)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("📊 Median ETA",  f"{np.median(etas_arr):.1f} min")
    c2.metric("📈 90th Pct",    f"{np.percentile(etas_arr,90):.1f} min")
    c3.metric("🔴 Expected Late", f"{(etas_arr>30).mean():.0%}")
    c4.metric("⚡ Best Case",   f"{np.percentile(etas_arr,10):.1f} min")

    fig_f = go.Figure()
    fig_f.add_trace(go.Histogram(
        x=etas_arr, nbinsx=30,
        marker_color="#6c8fff", opacity=0.8, name="ETA Distribution",
    ))
    fig_f.add_vline(x=30, line_dash="dot", line_color="#ff6b6b",
                    annotation_text="SLA 30 min", annotation_position="top right")
    fig_f.add_vline(x=np.median(etas_arr), line_dash="dash", line_color="#43d9ad",
                    annotation_text=f"Median {np.median(etas_arr):.1f}", annotation_position="top left")
    fig_f.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
        font_color="#e2e8f0", height=360,
        title=f"Expected ETA Distribution — {future_day} {future_hour} {WEATHER_PROFILES[future_wx]['label']}",
        xaxis_title="ETA (min)", yaxis_title="# Orders",
        margin=dict(t=40, b=20),
    )
    fig_f.update_xaxes(gridcolor="#2e3148")
    fig_f.update_yaxes(gridcolor="#2e3148")
    st.plotly_chart(fig_f, use_container_width=True)
