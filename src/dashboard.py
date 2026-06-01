"""
dashboard.py  –  Person 3
Plotly Dash dashboard with:
  1. Live map        – order locations on Bengaluru map (lat/lon simulated).
  2. Predicted vs Actual ETA scatter.
  3. Feature importance bar chart.
  4. Late delivery rate KPI + trend.

Run:
    python dashboard.py
Then open http://127.0.0.1:8050 in a browser.
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, callback

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = "Data"
MODEL_DIR = "Models"

# ── load artefacts ────────────────────────────────────────────────────────────

def load_predictions() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "predictions_test.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    # fallback: generate dummy data so dashboard launches without trained model
    n = 500
    rng = np.random.default_rng(42)
    true = rng.gamma(shape=5.0, scale=4.0, size=n)
    pred = true + rng.normal(0, 2.3, size=n)
    return pd.DataFrame({
        "y_true":       true.round(2),
        "y_pred":       pred.round(2),
        "abs_error":    np.abs(pred - true).round(2),
        "is_late_true": (true >= 30).astype(int),
        "is_late_pred": (pred >= 30).astype(int),
    })


def load_feature_importance() -> pd.DataFrame:
    path = os.path.join(MODEL_DIR, "feature_importance.csv")
    if os.path.exists(path):
        return pd.read_csv(path).head(15)
    # dummy
    features = [
        "ch_distance_km", "stage_delay_transit", "weather_mult",
        "stage_delay_prep", "tod_mult", "cpm_slack_min",
        "feat_distance_x_weather", "segment_count", "batch_size",
        "feat_transit_x_tod", "hour_of_day", "turn_count",
        "estimated_prep_variance", "day_of_week", "feat_prep_x_batch",
    ]
    rng = np.random.default_rng(0)
    scores = np.sort(rng.dirichlet(np.ones(len(features))))[::-1]
    return pd.DataFrame({"feature": features, "importance": scores.round(4)})


def load_eval_report() -> dict:
    path = os.path.join(MODEL_DIR, "evaluation_report.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def make_map_df(n: int = 300) -> pd.DataFrame:
    """
    Simulate order locations within Bengaluru bounding box.
    In production, replace with real lat/lon from order records.
    """
    rng    = np.random.default_rng(7)
    lat    = rng.uniform(12.85, 13.10, n)
    lon    = rng.uniform(77.50, 77.72, n)
    pred   = rng.gamma(shape=5, scale=4, size=n)
    true   = pred + rng.normal(0, 2, size=n)
    late   = (pred >= 30).astype(int)
    return pd.DataFrame({
        "lat":    lat.round(5),
        "lon":    lon.round(5),
        "pred_eta": pred.round(2),
        "true_eta": true.round(2),
        "status":   ["Late" if l else "On Time" for l in late],
    })


# ── colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "bg":        "#0f1117",
    "card":      "#1a1d27",
    "border":    "#2e3148",
    "primary":   "#6c8fff",
    "accent":    "#ff6b6b",
    "green":     "#43d9ad",
    "text":      "#e2e8f0",
    "muted":     "#8892a4",
    "on_time":   "#43d9ad",
    "late":      "#ff6b6b",
}

CARD_STYLE = {
    "backgroundColor": COLORS["card"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "12px",
    "padding": "20px",
    "marginBottom": "16px",
}

# ── build plots ───────────────────────────────────────────────────────────────

def fig_map(map_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter_mapbox(
        map_df,
        lat="lat", lon="lon",
        color="status",
        color_discrete_map={"On Time": COLORS["on_time"], "Late": COLORS["late"]},
        hover_data={"pred_eta": True, "true_eta": True, "lat": False, "lon": False},
        size_max=10,
        opacity=0.75,
        zoom=11,
        center={"lat": 12.97, "lon": 77.61},
        mapbox_style="carto-darkmatter",
        title="Live Order Map – Bengaluru",
    )
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font_color=COLORS["text"],
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title_text="",
        height=420,
    )
    return fig


def fig_scatter(pred_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    # perfect prediction line
    mn, mx = pred_df["y_true"].min(), pred_df["y_true"].max()
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines",
        line=dict(color=COLORS["muted"], dash="dash", width=1),
        name="Perfect prediction",
    ))
    # ±2.3 min bands
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn + 2.3, mx + 2.3],
        mode="lines", line=dict(color=COLORS["primary"], dash="dot", width=1),
        name="+2.3 min band", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn - 2.3, mx - 2.3],
        mode="lines", line=dict(color=COLORS["primary"], dash="dot", width=1),
        name="−2.3 min band", showlegend=False,
    ))
    # scatter
    fig.add_trace(go.Scatter(
        x=pred_df["y_true"], y=pred_df["y_pred"],
        mode="markers",
        marker=dict(
            color=pred_df["abs_error"],
            colorscale="RdYlGn_r",
            size=4, opacity=0.65,
            colorbar=dict(title="Abs Error (min)", thickness=12),
        ),
        name="Orders",
    ))
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["card"],
        font_color=COLORS["text"],
        xaxis_title="Actual ETA (min)",
        yaxis_title="Predicted ETA (min)",
        title="Predicted vs Actual ETA",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=380,
    )
    fig.update_xaxes(gridcolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"])
    return fig


def fig_importance(fi_df: pd.DataFrame) -> go.Figure:
    fi_df = fi_df.sort_values("importance")
    fig = go.Figure(go.Bar(
        x=fi_df["importance"],
        y=fi_df["feature"],
        orientation="h",
        marker=dict(
            color=fi_df["importance"],
            colorscale=[[0, COLORS["border"]], [1, COLORS["primary"]]],
        ),
    ))
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["card"],
        font_color=COLORS["text"],
        title="Feature Importance (Top 15)",
        xaxis_title="Importance Score",
        yaxis_title="",
        height=420,
        margin=dict(l=160, r=20, t=40, b=40),
    )
    fig.update_xaxes(gridcolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"])
    return fig


def fig_error_hist(pred_df: pd.DataFrame) -> go.Figure:
    errors = pred_df["y_pred"] - pred_df["y_true"]
    fig = go.Figure(go.Histogram(
        x=errors, nbinsx=50,
        marker_color=COLORS["primary"],
        opacity=0.8,
        name="Prediction error",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["muted"])
    fig.add_vline(x=2.3, line_dash="dot", line_color=COLORS["accent"], annotation_text="+2.3")
    fig.add_vline(x=-2.3, line_dash="dot", line_color=COLORS["accent"], annotation_text="−2.3")
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["card"],
        font_color=COLORS["text"],
        title="Prediction Error Distribution",
        xaxis_title="Error (min)  =  Pred − Actual",
        yaxis_title="Count",
        height=320,
    )
    fig.update_xaxes(gridcolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"])
    return fig


# ── KPI card helper ───────────────────────────────────────────────────────────

def kpi_card(label: str, value: str, color: str = COLORS["text"]) -> html.Div:
    return html.Div([
        html.P(label, style={"color": COLORS["muted"], "margin": "0", "fontSize": "13px"}),
        html.H3(value, style={"color": color, "margin": "4px 0 0 0", "fontSize": "26px"}),
    ], style={**CARD_STYLE, "textAlign": "center", "padding": "16px"})


# ── app layout ────────────────────────────────────────────────────────────────

def build_layout(pred_df: pd.DataFrame,
                 fi_df:   pd.DataFrame,
                 report:  dict,
                 map_df:  pd.DataFrame) -> html.Div:

    # KPIs from evaluation report
    test_reg  = report.get("test", {}).get("regression", {})
    test_late = report.get("test", {}).get("late_delivery", {})
    val_reg   = report.get("val",  {}).get("regression", {})

    mae      = val_reg.get("MAE", "–")
    late_rt  = test_late.get("late_rate_actual", pred_df["is_late_true"].mean())
    within   = val_reg.get("Within_2.3min_%", "–")
    r2       = test_reg.get("R2", "–")

    late_color = COLORS["accent"] if isinstance(late_rt, float) and late_rt > 0.15 else COLORS["green"]

    return html.Div(
        style={"backgroundColor": COLORS["bg"], "minHeight": "100vh",
               "fontFamily": "Inter, sans-serif", "padding": "24px"},
        children=[
            # ── header ────────────────────────────────────────────────────
            html.Div([
                html.H1("🛵  Bengaluru Delivery ETA Dashboard",
                        style={"color": COLORS["text"], "margin": "0",
                               "fontWeight": "700", "fontSize": "24px"}),
                html.P("Person 3  •  ML model + evaluation + live monitoring",
                       style={"color": COLORS["muted"], "margin": "4px 0 0 0",
                              "fontSize": "13px"}),
            ], style={"marginBottom": "24px"}),

            # ── KPI row ────────────────────────────────────────────────────
            html.Div([
                html.Div(kpi_card("Val MAE",
                                  f"{mae} min",
                                  COLORS["green"] if isinstance(mae, float) and mae <= 2.3
                                  else COLORS["accent"]),
                         style={"flex": "1", "marginRight": "12px"}),
                html.Div(kpi_card("Within ±2.3 min", f"{within}%", COLORS["primary"]),
                         style={"flex": "1", "marginRight": "12px"}),
                html.Div(kpi_card("Late Delivery Rate",
                                  f"{late_rt:.1%}" if isinstance(late_rt, float)
                                  else str(late_rt),
                                  late_color),
                         style={"flex": "1", "marginRight": "12px"}),
                html.Div(kpi_card("Test R²", str(r2), COLORS["primary"]),
                         style={"flex": "1"}),
            ], style={"display": "flex", "marginBottom": "8px"}),

            # ── map ────────────────────────────────────────────────────────
            html.Div([
                dcc.Graph(figure=fig_map(map_df), config={"displayModeBar": False}),
            ], style=CARD_STYLE),

            # ── scatter + importance ───────────────────────────────────────
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_scatter(pred_df),
                              config={"displayModeBar": False}),
                ], style={**CARD_STYLE, "flex": "1", "marginRight": "12px"}),
                html.Div([
                    dcc.Graph(figure=fig_importance(fi_df),
                              config={"displayModeBar": False}),
                ], style={**CARD_STYLE, "flex": "1"}),
            ], style={"display": "flex"}),

            # ── error distribution ─────────────────────────────────────────
            html.Div([
                dcc.Graph(figure=fig_error_hist(pred_df),
                          config={"displayModeBar": False}),
            ], style=CARD_STYLE),

            # ── footer ─────────────────────────────────────────────────────
            html.P(
                "Data simulated from Bengaluru OSM graph + DAG pipeline  •  "
                "Model: XGBoost GBDT  •  Target MAE ±2.3 min",
                style={"color": COLORS["muted"], "textAlign": "center",
                       "fontSize": "12px", "marginTop": "8px"},
            ),
        ]
    )


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pred_df = load_predictions()
    fi_df   = load_feature_importance()
    report  = load_eval_report()
    map_df  = make_map_df(300)

    app = dash.Dash(
        __name__,
        title="Bengaluru ETA Dashboard",
        external_stylesheets=[
            "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"
        ],
    )

    app.layout = build_layout(pred_df, fi_df, report, map_df)

    print("\n🚀  Dashboard running at  http://127.0.0.1:8050\n")
    app.run(debug=True, host="127.0.0.1", port=8050)