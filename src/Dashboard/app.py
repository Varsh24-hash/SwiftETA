"""
app.py  –  Person 3  |  Bengaluru Delivery Intelligence Dashboard
Entry point for the Streamlit multi-page app.

Run:
    streamlit run app.py

Pages (auto-discovered from pages/ folder):
    1_ETA_Predictor.py        Feature 1  – Real-Time ETA Predictor (main demo)
    2_Route_Visualizer.py     Feature 2  – Live Route Visualisation
    3_ETA_Breakdown.py        Feature 3  – ETA Breakdown
    4_WhatIf_Simulator.py     Feature 4  – What-If Simulator
    5_Delay_Risk_Meter.py     Feature 5  – Delay Risk Meter
    6_Model_Explainer.py      Feature 6  – Why Did The Model Predict This?
    7_Restaurant_Rec.py       Feature 7  – Best Restaurant Recommendation
    8_Peak_Analytics.py       Feature 8  – Peak-Hour Analytics
    9_Fleet_Panel.py          Feature 9+10 – Fleet Intelligence + Bottleneck
   10_Heatmap_Forecast.py     Feature 11+12 – Delivery Heatmap + ETA Forecast
   11_Scorecard.py            Feature 13 – Delivery Scorecard
   12_AI_Assistant.py         Feature 14 – AI Assistant
   13_Comparison_Mode.py      Feature 15 – Live Comparison Mode
"""

import streamlit as st

st.set_page_config(
    page_title="Bengaluru Delivery Intelligence",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0f1117; }
  .stMetric label          { font-size: 13px !important; color: #8892a4 !important; }
  .stMetric value          { font-size: 26px !important; }
  div[data-testid="metric-container"] {
      background: #1a1d27;
      border: 1px solid #2e3148;
      border-radius: 10px;
      padding: 14px 18px;
  }
  .hero-title {
      font-size: 2.4rem; font-weight: 800;
      background: linear-gradient(90deg, #6c8fff, #43d9ad);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hero-sub { color: #8892a4; font-size: 1rem; margin-top: -8px; }
  .feature-card {
      background: #1a1d27; border: 1px solid #2e3148;
      border-radius: 12px; padding: 20px; margin-bottom: 12px;
  }
  .tag {
      display: inline-block; padding: 2px 10px;
      border-radius: 999px; font-size: 12px; font-weight: 600;
      margin-right: 6px;
  }
  .tag-p1 { background: #2a1f5e; color: #6c8fff; }
  .tag-p2 { background: #1a3529; color: #43d9ad; }
  .tag-p3 { background: #3a2010; color: #ffb347; }
</style>
""", unsafe_allow_html=True)

# ── landing page ──────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🛵 Bengaluru Delivery Intelligence</div>',
            unsafe_allow_html=True)
st.markdown('<div class="hero-sub">ML-powered ETA prediction • Built by P1 + P2 + P3</div>',
            unsafe_allow_html=True)

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="feature-card">
      <b>Person 1 – Graph Engine</b><br>
      <span class="tag tag-p1">Contraction Hierarchies</span>
      <span class="tag tag-p1">Dijkstra</span><br><br>
      Loads Bengaluru OSM graph, runs CH preprocessing and bidirectional
      Dijkstra queries to extract route distance, segment count, turn count
      and road-type fractions.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
      <b>Person 2 – DAG Scheduler</b><br>
      <span class="tag tag-p2">Kahn's Sort</span>
      <span class="tag tag-p2">CPM</span>
      <span class="tag tag-p2">Greedy Batching</span><br><br>
      Models the delivery lifecycle as a DAG, runs CPM to find the critical
      path and computes cpm_slack, stage_delays, batch_size and prep variance.
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
      <b>Person 3 – ML Model + Dashboard</b><br>
      <span class="tag tag-p3">XGBoost</span>
      <span class="tag tag-p3">Feature Engineering</span>
      <span class="tag tag-p3">Plotly</span><br><br>
      Trains XGBoost on P1+P2 features, evaluates with MAE ±2.3 min target,
      and serves predictions through this 15-feature Streamlit dashboard.
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("### 📋 Dashboard Pages")

pages = [
    ("🚀", "1 · ETA Predictor",        "Real-time prediction from route + DAG + weather"),
    ("🗺️", "2 · Route Visualizer",      "Live route on Bengaluru map with P1 stats"),
    ("⚙️", "3 · ETA Breakdown",         "Full decomposition: prep + pickup + transit + weather"),
    ("🌧️", "4 · What-If Simulator",     "Sliders for weather, hour, batch size, traffic"),
    ("🔥", "5 · Delay Risk Meter",       "Fuel-gauge style late-delivery probability"),
    ("🧠", "6 · Model Explainer",        "Feature importance → why this ETA?"),
    ("🏆", "7 · Restaurant Recommender","Compare areas by ETA, recommend fastest"),
    ("📈", "8 · Peak-Hour Analytics",    "Hour-by-hour average ETA chart"),
    ("🚛", "9 · Fleet Intelligence",     "Ops dashboard + bottleneck detection (CPM)"),
    ("🎯", "10 · Heatmap + Forecast",    "Delivery heatmap + future ETA distribution"),
    ("🥇", "11 · Delivery Scorecard",    "Route / Traffic / Weather / Ops scores"),
    ("🤖", "12 · AI Assistant",          "Natural-language ETA explanation"),
    ("⭐", "13 · Live Comparison",       "Side-by-side scenario comparison"),
]

for emoji, name, desc in pages:
    st.markdown(f"**{emoji} {name}** — {desc}")

st.divider()
st.caption("👈 Use the sidebar to navigate between pages.")
