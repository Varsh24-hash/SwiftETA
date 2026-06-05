"""pages/13_Comparison_Mode.py – Feature 15: Live Comparison Mode"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import WEATHER_PROFILES, tod_mult, run_p2, compute_route_time, compute_eta, delay_risk, scorecard, PROMISED_ETA

st.set_page_config(page_title="Live Comparison", page_icon="⭐", layout="wide")

st.markdown("# ⭐ Live Comparison Mode")
st.caption("Feature 15 — Side-by-side scenario comparison  |  ETA · Delay Risk · Scorecard")
st.divider()

ROAD_PROFILES = {
    "Koramangala":  {"km":4.2,"hw":0.10,"pr":0.40,"se":0.30,"re":0.20},
    "HSR Layout":   {"km":5.8,"hw":0.15,"pr":0.35,"se":0.30,"re":0.20},
    "Indiranagar":  {"km":6.5,"hw":0.08,"pr":0.32,"se":0.38,"re":0.22},
    "Whitefield":   {"km":12.1,"hw":0.25,"pr":0.38,"se":0.22,"re":0.15},
    "Marathahalli": {"km":9.4,"hw":0.20,"pr":0.40,"se":0.25,"re":0.15},
    "Electronic City":{"km":14.2,"hw":0.40,"pr":0.35,"se":0.15,"re":0.10},
}

st.markdown("### Configure Scenarios")
col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown("#### 🅰️ Scenario A")
    area_a  = st.selectbox("📍 Area",    list(ROAD_PROFILES.keys()), index=0, key="aa")
    wx_a    = st.selectbox("🌤️ Weather", list(WEATHER_PROFILES.keys()), index=0, key="wa",
                            format_func=lambda k: WEATHER_PROFILES[k]["label"])
    hour_a  = st.slider("🕐 Hour", 0, 23, 10, key="ha")
    batch_a = st.slider("📦 Batch", 1, 5, 1, key="ba")
    seed_a  = st.number_input("🔢 Seed", 1, 9999, 42, key="sa")

with col_b:
    st.markdown("#### 🅱️ Scenario B")
    area_b  = st.selectbox("📍 Area",    list(ROAD_PROFILES.keys()), index=3, key="ab")
    wx_b    = st.selectbox("🌤️ Weather", list(WEATHER_PROFILES.keys()), index=3, key="wb",
                            format_func=lambda k: WEATHER_PROFILES[k]["label"])
    hour_b  = st.slider("🕐 Hour", 0, 23, 18, key="hb")
    batch_b = st.slider("📦 Batch", 1, 5, 4, key="bb")
    seed_b  = st.number_input("🔢 Seed", 1, 9999, 99, key="sb")

def compute_scenario(area, wx_k, hour, batch_sz, seed):
    rp     = ROAD_PROFILES[area]
    wm     = WEATHER_PROFILES[wx_k]["mult"]
    tm     = tod_mult(hour)
    p2     = run_p2(seed=int(seed), n_window=batch_sz)
    rt     = compute_route_time(rp["km"], rp["hw"], rp["pr"], rp["se"], rp["re"], wm, tm)
    wd     = round(rt * (wm - 1.0), 2)
    eta    = compute_eta(rt, p2["cpm_eta"], wd)
    rl, rc, rp_val = delay_risk(eta)
    sc     = scorecard(rp["km"], wm, tm, batch_sz, p2["cpm_slack"])
    return {
        "eta": eta, "risk": rl, "risk_color": rc, "late_prob": rp_val,
        "route_time": rt, "weather_delay": wd, "cpm_eta": p2["cpm_eta"],
        "cpm_slack": p2["cpm_slack"], "batch_size": p2["batch_size"],
        "scores": sc, "critical_path": p2["critical_path"],
        "is_late": eta > PROMISED_ETA,
        "stage_delays": p2["stage_delays"],
    }

sc_a = compute_scenario(area_a, wx_a, hour_a, batch_a, seed_a)
sc_b = compute_scenario(area_b, wx_b, hour_b, batch_b, seed_b)

st.divider()
st.markdown("### 📊 Side-by-Side Results")

# ── metric comparison table ───────────────────────────────────────────────────
metrics = [
    ("⏱️ Predicted ETA",       f"{sc_a['eta']} min",           f"{sc_b['eta']} min"),
    ("⚠️ Delay Risk",          sc_a["risk"],                    sc_b["risk"]),
    ("🎲 Late Probability",    f"{sc_a['late_prob']:.0%}",      f"{sc_b['late_prob']:.0%}"),
    ("🛣️ Route Time (P1)",     f"{sc_a['route_time']} min",     f"{sc_b['route_time']} min"),
    ("⚙️ CPM ETA (P2)",        f"{sc_a['cpm_eta']} min",        f"{sc_b['cpm_eta']} min"),
    ("🌧️ Weather Delay",       f"{sc_a['weather_delay']} min",  f"{sc_b['weather_delay']} min"),
    ("📦 CPM Slack (P2)",      f"{sc_a['cpm_slack']} min",      f"{sc_b['cpm_slack']} min"),
    ("🥇 Overall Score",       f"{sc_a['scores']['Overall']}/100", f"{sc_b['scores']['Overall']}/100"),
    ("🟢 SLA Status",          "🔴 LATE" if sc_a["is_late"] else "🟢 ON TIME",
                                "🔴 LATE" if sc_b["is_late"] else "🟢 ON TIME"),
]

def color_cell(val_a, val_b, metric_name):
    """Highlight winner in green"""
    if "ETA" in metric_name or "Delay" in metric_name or "Prob" in metric_name or "Late" in metric_name or "Risk" in metric_name:
        # lower is better
        try:
            fa = float(str(val_a).replace(" min","").replace("%","").replace("/100",""))
            fb = float(str(val_b).replace(" min","").replace("%","").replace("/100",""))
            return ("🏆 " + val_a if fa < fb else val_a), ("🏆 " + val_b if fb < fa else val_b)
        except: pass
    if "Score" in metric_name:
        try:
            fa = float(str(val_a).split("/")[0])
            fb = float(str(val_b).split("/")[0])
            return ("🏆 " + val_a if fa > fb else val_a), ("🏆 " + val_b if fb > fa else val_b)
        except: pass
    return val_a, val_b

rows = []
for m, va, vb in metrics:
    va2, vb2 = color_cell(va, vb, m)
    rows.append({"Metric": m, "Scenario A": va2, "Scenario B": vb2})

cmp_df = pd.DataFrame(rows)
st.dataframe(cmp_df, use_container_width=True, hide_index=True)

# ── radar comparison ──────────────────────────────────────────────────────────
st.divider()
st.markdown("### 🕸️ Scorecard Radar Comparison")
cats = ["Route Efficiency","Traffic Impact","Weather Impact","Operational"]

fig_radar = go.Figure()
for label, sc, color in [("Scenario A", sc_a, "#43d9ad"), ("Scenario B", sc_b, "#ff6b6b")]:
    vals = [sc["scores"][c] for c in cats] + [sc["scores"][cats[0]]]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals, theta=cats + [cats[0]],
        fill="toself", fillcolor=color.replace(")", ",0.12)").replace("rgb","rgba") if "rgb" in color else color+"26",
        line=dict(color=color, width=2),
        name=label,
    ))
fig_radar.update_layout(
    polar=dict(
        bgcolor="#1a1d27",
        radialaxis=dict(visible=True, range=[0,100], gridcolor="#2e3148", linecolor="#2e3148"),
        angularaxis=dict(gridcolor="#2e3148", linecolor="#2e3148"),
    ),
    paper_bgcolor="#0f1117", font_color="#e2e8f0",
    height=420, legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=20,b=20,l=60,r=60),
)
st.plotly_chart(fig_radar, use_container_width=True)

# ── ETA bar comparison ────────────────────────────────────────────────────────
st.markdown("### 📊 ETA Breakdown Comparison")
components = ["Route Time","CPM ETA","Weather Delay"]
vals_a = [sc_a["route_time"], sc_a["cpm_eta"], sc_a["weather_delay"]]
vals_b = [sc_b["route_time"], sc_b["cpm_eta"], sc_b["weather_delay"]]

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(name="Scenario A", x=components, y=vals_a, marker_color="#43d9ad",
                          text=[f"{v} min" for v in vals_a], textposition="outside"))
fig_bar.add_trace(go.Bar(name="Scenario B", x=components, y=vals_b, marker_color="#ff6b6b",
                          text=[f"{v} min" for v in vals_b], textposition="outside"))
fig_bar.update_layout(
    paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
    font_color="#e2e8f0", barmode="group", height=340,
    yaxis_title="Minutes", margin=dict(t=20,b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)
fig_bar.update_xaxes(gridcolor="#2e3148")
fig_bar.update_yaxes(gridcolor="#2e3148")
st.plotly_chart(fig_bar, use_container_width=True)

# ── winner summary ────────────────────────────────────────────────────────────
st.divider()
winner = "A" if sc_a["eta"] < sc_b["eta"] else "B"
diff   = abs(sc_a["eta"] - sc_b["eta"])
st.success(f"✅ **Scenario {winner} wins** — {diff:.1f} min faster ETA  |  "
           f"Score: A={sc_a['scores']['Overall']}/100  B={sc_b['scores']['Overall']}/100")
