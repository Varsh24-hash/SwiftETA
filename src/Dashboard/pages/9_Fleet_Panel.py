"""pages/9_Fleet_Panel.py – Feature 9: Fleet Intelligence Panel + Feature 10: Bottleneck Detection"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import run_p2, WEATHER_PROFILES, tod_mult, compute_route_time, compute_eta, load_all_orders, PROMISED_ETA

st.set_page_config(page_title="Fleet Intelligence", page_icon="🚛", layout="wide")

st.markdown("# Fleet Intelligence Panel")
st.caption("Feature 9 + 10 — Operations dashboard + Bottleneck Detection using P2 CPM results")
st.divider()

with st.sidebar:
    st.markdown("### 🎛️ Fleet Parameters")
    n_orders  = st.slider("📦 Orders in Window", 10, 100, 40)
    weather_k = st.selectbox("🌤️ Weather", list(WEATHER_PROFILES.keys()),
                              format_func=lambda k: WEATHER_PROFILES[k]["label"])
    hour      = st.slider("🕐 Current Hour", 0, 23, 18)
    seed_base = st.number_input("🔢 Base Seed", 1, 9999, 42)

AREA_KM = {"Koramangala":4.2,"HSR Layout":5.8,"Indiranagar":6.5,
            "Whitefield":12.1,"Marathahalli":9.4,"Jayanagar":7.3}
AREA_PROFILES = {
    "Koramangala":  {"hw":0.10,"pr":0.40,"se":0.30,"re":0.20},
    "HSR Layout":   {"hw":0.15,"pr":0.35,"se":0.30,"re":0.20},
    "Indiranagar":  {"hw":0.08,"pr":0.32,"se":0.38,"re":0.22},
    "Whitefield":   {"hw":0.25,"pr":0.38,"se":0.22,"re":0.15},
    "Marathahalli": {"hw":0.20,"pr":0.40,"se":0.25,"re":0.15},
    "Jayanagar":    {"hw":0.05,"pr":0.28,"se":0.42,"re":0.25},
}

areas_list = list(AREA_KM.keys())
rng_fleet  = np.random.default_rng(int(seed_base))
w_mult     = WEATHER_PROFILES[weather_k]["mult"]
t_mult     = tod_mult(hour)

# generate fleet orders from real P2 + P1-proxy
orders_data = []
for i in range(n_orders):
    area  = areas_list[i % len(areas_list)]
    ap    = AREA_PROFILES[area]
    km    = AREA_KM[area]
    p2    = run_p2(seed=int(seed_base)+i)
    rt    = compute_route_time(km, ap["hw"], ap["pr"], ap["se"], ap["re"], w_mult, t_mult)
    wd    = rt * (w_mult - 1.0)
    eta   = compute_eta(rt, p2["cpm_eta"], wd, float(rng_fleet.normal(0,1)))
    orders_data.append({
        "order_id":  f"ORD-{1000+i}",
        "area":      area,
        "eta":       eta,
        "is_late":   int(eta > PROMISED_ETA),
        "batch_sz":  p2["batch_size"],
        "distance":  km,
        "cpm_slack": p2["cpm_slack"],
        "stage_prep":    p2["stage_delays"].get("accepted_to_prep",0),
        "stage_pickup":  p2["stage_delays"].get("prep_to_pickup",0),
        "stage_transit": p2["stage_delays"].get("pickup_to_transit",0),
    })

fleet_df = pd.DataFrame(orders_data)

# ── KPI row ───────────────────────────────────────────────────────────────────
avg_batch    = round(fleet_df["batch_sz"].mean(), 2)
avg_dist     = round(fleet_df["distance"].mean(), 2)
avg_eta      = round(fleet_df["eta"].mean(), 2)
late_count   = fleet_df["is_late"].sum()
late_pct     = round(late_count / n_orders * 100, 1)

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("📦 Avg Batch Size",    avg_batch)
c2.metric("📏 Avg Distance",       f"{avg_dist} km")
c3.metric("⏱️ Avg ETA",            f"{avg_eta} min")
c4.metric("🔴 Late Deliveries",    late_count, delta=f"{late_pct}% of fleet",
          delta_color="inverse")
c5.metric("🟢 On-Time Rate",       f"{100-late_pct}%")

st.divider()

col_map, col_breakdown = st.columns([1,1], gap="large")

with col_map:
    st.markdown("#### 📊 ETA Distribution by Area")
    area_avg = fleet_df.groupby("area")["eta"].mean().reset_index().sort_values("eta")
    fig_area = go.Figure(go.Bar(
        x=area_avg["eta"].round(1), y=area_avg["area"],
        orientation="h",
        marker_color=["#ff6b6b" if e>30 else "#43d9ad" for e in area_avg["eta"]],
        text=[f"{e:.1f} min" for e in area_avg["eta"]],
        textposition="outside",
    ))
    fig_area.add_vline(x=30, line_dash="dot", line_color="#ff6b6b")
    fig_area.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                           font_color="#e2e8f0", height=320, showlegend=False,
                           margin=dict(l=110, t=10, b=10, r=60))
    fig_area.update_xaxes(gridcolor="#2e3148")
    fig_area.update_yaxes(gridcolor="#2e3148")
    st.plotly_chart(fig_area, use_container_width=True)

with col_breakdown:
    st.markdown("#### 📦 Batch Size Distribution")
    batch_counts = fleet_df["batch_sz"].value_counts().reset_index()
    batch_counts.columns = ["Batch Size","Count"]
    fig_batch = px.pie(batch_counts, values="Count", names="Batch Size",
                       color_discrete_sequence=["#6c8fff","#43d9ad","#f5a623","#ff6b6b","#a78bfa"])
    fig_batch.update_layout(paper_bgcolor="#0f1117", font_color="#e2e8f0",
                             height=320, margin=dict(t=10,b=10))
    st.plotly_chart(fig_batch, use_container_width=True)

# ── Feature 10: Bottleneck Detection ─────────────────────────────────────────
st.divider()
st.markdown("## ⚠️ Feature 10: Bottleneck Detection")
st.caption("Using CPM stage_delays from P2 — which stage contributes most delay?")

total_prep    = fleet_df["stage_prep"].sum()
total_pickup  = fleet_df["stage_pickup"].sum()
total_transit = fleet_df["stage_transit"].sum()
grand_total   = total_prep + total_pickup + total_transit

bottleneck_data = pd.DataFrame({
    "Stage":      ["Prep Stage", "Transit Stage", "Pickup Stage"],
    "Total Delay":[total_prep, total_transit, total_pickup],
    "Percentage": [
        round(total_prep/grand_total*100,1),
        round(total_transit/grand_total*100,1),
        round(total_pickup/grand_total*100,1),
    ],
}).sort_values("Percentage", ascending=False)

worst_stage = bottleneck_data.iloc[0]["Stage"]
worst_pct   = bottleneck_data.iloc[0]["Percentage"]
st.error(f"🚨 **Biggest Delay Source:** {worst_stage} — {worst_pct}% of total stage delay")

col_btn, col_bpie = st.columns([1,1], gap="large")
with col_btn:
    st.markdown("**Stage Delay Breakdown:**")
    for _, row in bottleneck_data.iterrows():
        bar = "█" * int(row["Percentage"]/5) + "░" * (20 - int(row["Percentage"]/5))
        st.markdown(
            f"`{row['Stage']:<20}` "
            f"<span style='color:#6c8fff; font-family:monospace'>[{bar}] {row['Percentage']}%</span>",
            unsafe_allow_html=True,
        )
with col_bpie:
    fig_b = px.pie(bottleneck_data, values="Percentage", names="Stage",
                   color_discrete_sequence=["#ff6b6b","#f5a623","#43d9ad"],
                   hole=0.4)
    fig_b.update_layout(paper_bgcolor="#0f1117", font_color="#e2e8f0",
                         height=300, margin=dict(t=10,b=10))
    st.plotly_chart(fig_b, use_container_width=True)

st.divider()
st.markdown("**Critical Path (P2 CPM):**")
p2_sample = run_p2(seed=int(seed_base))
cp_str = " → ".join(p2_sample["critical_path"])
st.code(cp_str, language=None)
slack_df = pd.DataFrame({
    "Stage": list(p2_sample["slack_per_stage"].keys()),
    "Slack (min)": list(p2_sample["slack_per_stage"].values()),
})
slack_df["On Critical Path"] = slack_df["Slack (min)"].apply(lambda x: "✅ Yes" if x==0 else "")
st.dataframe(slack_df, use_container_width=True, hide_index=True)
