"""
feature_engineering.py  –  Person 3
Builds the final ML-ready feature matrix from:
  • P1 graph engine outputs  (CH distance, road type, segment/turn counts)
  • P2 DAG outputs           (CPM slack, batch size, prep variance, stage delays)
  • Temporal / weather       (hour, DOW, weather code)
  • Target                   ground_truth_eta  (regression) + is_late (classification)
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = "Data"
MODEL_DIR = "Models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── feature groups (mirrors image spec exactly) ───────────────────────────────

P1_GRAPH_FEATURES = [
    "ch_distance_km",
    "road_highway_frac",
    "road_primary_frac",
    "road_secondary_frac",
    "road_residential_frac",
    "segment_count",
    "turn_count",
]

P2_DAG_FEATURES = [
    "cpm_slack_min",
    "batch_size",
    "estimated_prep_variance",
    "stage_delay_placed",
    "stage_delay_prep",
    "stage_delay_pickup",
    "stage_delay_transit",
    "stage_delay_delivered",
]

TEMPORAL_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "weather_mult",
    "tod_mult",
]

TARGET_COL      = "ground_truth_eta"
CLASS_TARGET    = "is_late"

ALL_FEATURES    = P1_GRAPH_FEATURES + P2_DAG_FEATURES + TEMPORAL_FEATURES


# ── helpers ───────────────────────────────────────────────────────────────────

def encode_weather(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the weather column (drop first to avoid multicollinearity)."""
    if "weather" in df.columns:
        dummies = pd.get_dummies(df["weather"], prefix="weather", drop_first=True)
        df = pd.concat([df, dummies], axis=1)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-driven interaction terms.
    • distance_x_weather  – longer route + bad weather compounds delay.
    • transit_x_tod       – transit stage hit by time-of-day congestion.
    • prep_x_batch        – larger batch → higher prep pressure.
    """
    df = df.copy()
    df["feat_distance_x_weather"] = (
        df["ch_distance_km"] * df["weather_mult"]
    ).round(4)
    df["feat_transit_x_tod"] = (
        df["stage_delay_transit"] * df["tod_mult"]
    ).round(4)
    df["feat_prep_x_batch"] = (
        df["stage_delay_prep"] * df["batch_size"]
    ).round(4)
    df["feat_cpm_per_km"] = (
        df["cpm_slack_min"] / df["ch_distance_km"].clip(lower=0.1)
    ).round(4)
    return df


def build_feature_matrix(df: pd.DataFrame,
                          scaler: StandardScaler | None = None,
                          fit_scaler: bool = False
                          ) -> tuple[pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Parameters
    ----------
    df          : raw DataFrame from data_collector (train / val / test)
    scaler      : pre-fitted StandardScaler (pass None for train split)
    fit_scaler  : True  → fit a new scaler on df  (use for train only)

    Returns
    -------
    X_scaled    : scaled feature DataFrame
    y_reg       : regression target (ground_truth_eta)
    y_cls       : classification target (is_late)
    scaler      : fitted StandardScaler
    """
    df = encode_weather(df)
    df = add_interaction_features(df)

    # collect all features that exist in this df
    weather_dummies = [c for c in df.columns if c.startswith("weather_")]
    interaction_cols = [c for c in df.columns if c.startswith("feat_")]
    feature_cols = ALL_FEATURES + weather_dummies + interaction_cols

    # keep only columns that are present (robustness)
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy().fillna(0.0)

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=feature_cols,
            index=X.index
        )
    else:
        if scaler is None:
            raise ValueError("Pass a fitted scaler for val/test splits.")
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=feature_cols,
            index=X.index
        )

    y_reg = df[TARGET_COL].reset_index(drop=True)
    y_cls = df[CLASS_TARGET].reset_index(drop=True)

    return X_scaled, y_reg, y_cls, scaler


def summarise(X: pd.DataFrame, y: pd.Series, split_name: str) -> None:
    print(f"\n── {split_name} ──")
    print(f"  Samples   : {len(X)}")
    print(f"  Features  : {X.shape[1]}")
    print(f"  ETA  mean : {y.mean():.2f} min   std : {y.std():.2f} min")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading splits …")
    train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                            parse_dates=["timestamp"])
    val_raw   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"),
                            parse_dates=["timestamp"])
    test_raw  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                            parse_dates=["timestamp"])

    print("Engineering features …")
    X_train, y_train, y_train_cls, scaler = build_feature_matrix(
        train_raw, fit_scaler=True)
    X_val,   y_val,   y_val_cls,   _      = build_feature_matrix(
        val_raw,   scaler=scaler)
    X_test,  y_test,  y_test_cls,  _      = build_feature_matrix(
        test_raw,  scaler=scaler)

    summarise(X_train, y_train, "TRAIN")
    summarise(X_val,   y_val,   "VAL")
    summarise(X_test,  y_test,  "TEST")

    # ── save engineered splits ─────────────────────────────────────────────
    for split_name, X, y_r, y_c in [
        ("train", X_train, y_train, y_train_cls),
        ("val",   X_val,   y_val,   y_val_cls),
        ("test",  X_test,  y_test,  y_test_cls),
    ]:
        out = X.copy()
        out[TARGET_COL] = y_r.values
        out[CLASS_TARGET] = y_c.values
        out.to_csv(os.path.join(DATA_DIR, f"{split_name}_features.csv"), index=False)

    # ── persist scaler ─────────────────────────────────────────────────────
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    print(f"\nScaler saved  → Models/scaler.pkl")
    print("Feature CSVs saved → Data/{{train|val|test}}_features.csv")
    print("\nFeature list:")
    for col in X_train.columns:
        print(f"  {col}")