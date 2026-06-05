"""
engine.py  –  Shared computation layer for the dashboard.
All pages import from here so P1/P2/P3 logic is called in one place.

Folder layout:
    SwiftETA/
      src/
        dag_builder.py      ← P1/P2 files live here
        kahn_sort.py
        stochastic_weights.py
        cpm.py
        batch_scheduler.py
        Dashboard/
          app.py
          engine.py         ← this file
          pages/
            1_ETA_Predictor.py
            ...
"""

import os, sys, random
import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
# Dashboard/  →  engine.py lives here
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
# src/        →  one level up, where dag_builder.py etc. live
SRC_DIR = os.path.dirname(DASHBOARD_DIR)

for p in [DASHBOARD_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── P2 imports (files are in src/) ───────────────────────────────────────────
from dag_builder        import build_dag, STAGES
from kahn_sort          import kahns_sort
from stochastic_weights import sample_durations
from cpm                import run_cpm
from batch_scheduler    import greedy_batch_scheduler

# ── constants ─────────────────────────────────────────────────────────────────
WEATHER_PROFILES = {
    "clear":      {"mult": 1.00, "label": "☀️ Clear"},
    "cloudy":     {"mult": 1.05, "label": "☁️ Cloudy"},
    "rainy":      {"mult": 1.25, "label": "🌧️ Rainy"},
    "heavy_rain": {"mult": 1.55, "label": "⛈️ Heavy Rain"},
}
ROAD_SPEED   = {"highway": 1.0, "primary": 0.80, "secondary": 0.65, "residential": 0.50}
AVG_SPEED    = 25.0   # km/h base urban speed
PROMISED_ETA = 30.0   # SLA minutes

# pre-build DAG once at import time
_ADJ, _IN_DEG, _EDGE_DATA = build_dag()
_TOPO = kahns_sort(_ADJ, _IN_DEG)

# ── TOD multiplier ────────────────────────────────────────────────────────────
def tod_mult(hour: int) -> float:
    if 8 <= hour <= 10 or 17 <= hour <= 20: return 1.35
    if 11 <= hour <= 14:                    return 1.10
    return 1.0

# ── P2: run full DAG pipeline for one order ───────────────────────────────────
def run_p2(seed: int = 42, n_window: int = 3) -> dict:
    sampled, prep_var = sample_durations(_EDGE_DATA, seed=seed)
    cpm = run_cpm(_TOPO, _ADJ, _EDGE_DATA, sampled)

    rng = np.random.default_rng(seed)
    eta = cpm["total_eta_minutes"]
    window = [(f"O{k}", round(float(rng.uniform(0, 5)), 1),
               round(float(rng.uniform(eta * 0.8, eta * 1.2)), 1))
              for k in range(max(n_window, 2))]
    batch = greedy_batch_scheduler(window)

    return {
        "cpm_eta":         round(eta, 2),
        "cpm_slack":       round(cpm["cpm_slack"], 2),
        "critical_path":   cpm["critical_path"],
        "stage_delays":    cpm["stage_delays"],
        "slack_per_stage": cpm["slack"],
        "prep_variance":   round(prep_var, 4),
        "batch_size":      batch["batch_size"],
        "sampled":         sampled,
    }

# ── compute route time from P1 features ──────────────────────────────────────
def compute_route_time(distance_km, road_hw, road_pr, road_se, road_re,
                       w_mult, t_mult) -> float:
    ws = (road_hw * ROAD_SPEED["highway"]
        + road_pr * ROAD_SPEED["primary"]
        + road_se * ROAD_SPEED["secondary"]
        + road_re * ROAD_SPEED["residential"]) * AVG_SPEED
    speed = max(ws / (w_mult * t_mult), 1.0)
    return round((distance_km / speed) * 60, 2)

# ── ground-truth ETA formula ──────────────────────────────────────────────────
def compute_eta(route_time, cpm_eta, weather_delay, noise=0.0) -> float:
    return round(max(route_time + cpm_eta + weather_delay + noise, 1.0), 2)

# ── delay risk ────────────────────────────────────────────────────────────────
def delay_risk(eta: float) -> tuple:
    prob = min(max((eta - 20) / 20, 0.0), 1.0)
    if prob < 0.35: return "LOW",    "#43d9ad", prob
    if prob < 0.65: return "MEDIUM", "#f5a623", prob
    return                 "HIGH",   "#ff6b6b", prob

# ── delivery scorecard ────────────────────────────────────────────────────────
def scorecard(distance_km, w_mult, t_mult, batch_size, cpm_slack) -> dict:
    route_score   = max(0, 100 - int(distance_km * 4))
    traffic_score = max(0, 100 - int((t_mult - 1) * 200))
    weather_score = max(0, 100 - int((w_mult - 1) * 250))
    ops_score     = min(100, int(50 + cpm_slack * 5 - batch_size * 3))
    overall       = round((route_score + traffic_score + weather_score + ops_score) / 4)
    return {
        "Route Efficiency": route_score,
        "Traffic Impact":   traffic_score,
        "Weather Impact":   weather_score,
        "Operational":      ops_score,
        "Overall":          overall,
    }

# ── per-prediction explainer ──────────────────────────────────────────────────
def explain_eta(route_time, cpm_eta, weather_delay, t_mult, batch_size) -> list:
    items = [
        ("Travel Time (route)",      round(route_time, 1),           "+"),
        ("Food Prep + Stages (CPM)", round(cpm_eta, 1),              "+"),
        ("Weather Delay",            round(weather_delay, 1),        "+" if weather_delay > 0 else "="),
        ("Peak-Hour Traffic",        round((t_mult - 1) * route_time, 1), "+" if t_mult > 1 else "="),
        ("Batch Size Impact",        round((batch_size - 1) * 0.8, 1),   "+" if batch_size > 1 else "="),
    ]
    return [{"factor": f, "delta": d, "sign": s} for f, d, s in items]

# ── load trained model + scaler (graceful fallback if not yet trained) ────────
import joblib

def load_model():
    mp = os.path.join(DASHBOARD_DIR, "Models", "xgboost_eta.pkl")
    sp = os.path.join(DASHBOARD_DIR, "Models", "scaler.pkl")
    if os.path.exists(mp) and os.path.exists(sp):
        return joblib.load(mp), joblib.load(sp)
    return None, None

def load_feature_importance() -> pd.DataFrame:
    fp = os.path.join(DASHBOARD_DIR, "Models", "feature_importance.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp)
    return pd.DataFrame()

def load_test_predictions() -> pd.DataFrame:
    fp = os.path.join(DASHBOARD_DIR, "Data", "predictions_test.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp)
    return pd.DataFrame()

def load_all_orders() -> pd.DataFrame:
    fp = os.path.join(DASHBOARD_DIR, "Data", "all_orders.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp, parse_dates=["timestamp"])
    return pd.DataFrame()