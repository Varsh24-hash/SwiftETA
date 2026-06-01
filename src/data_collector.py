"""
data_collector.py  –  Person 3
Simulate historical delivery data with ground-truth ETAs.
Produces train / val / test splits (by date) and saves them as CSV files.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = "Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── simulation constants ──────────────────────────────────────────────────────
N_ORDERS          = 5_000          # total simulated orders
START_DATE        = datetime(2024, 1, 1)
END_DATE          = datetime(2024, 12, 31)

# Stage base durations (minutes) – mirrored from dag_builder.py EDGES
STAGE_PARAMS = {
    "placed_to_accepted":  {"base": 1.5,  "shape": 3.0,  "scale": 0.5},
    "accepted_to_prep":    {"base": 8.0,  "shape": 4.0,  "scale": 2.0},
    "prep_to_pickup":      {"base": 3.0,  "shape": 3.0,  "scale": 1.0},
    "pickup_to_transit":   {"base": 12.0, "shape": 6.0,  "scale": 2.0},
    "transit_to_delivered":{"base": 0.5,  "shape": 2.0,  "scale": 0.25},
}

# Weather impact multipliers (weather → speed factor on travel time)
WEATHER_PROFILES = {
    "clear":   1.0,
    "cloudy":  1.05,
    "rainy":   1.25,
    "heavy_rain": 1.55,
}

WEATHER_PROBS = [0.50, 0.25, 0.18, 0.07]   # must sum to 1

# Road-type distribution (fraction of segments)
ROAD_TYPES = ["highway", "primary", "secondary", "residential"]

# ── helper functions ──────────────────────────────────────────────────────────

def random_timestamp(start: datetime, end: datetime) -> datetime:
    delta = (end - start).total_seconds()
    return start + timedelta(seconds=random.uniform(0, delta))


def time_of_day_multiplier(hour: int) -> float:
    """Peak-hour slow-down on transit/pickup stages."""
    if 8 <= hour <= 10 or 17 <= hour <= 20:
        return 1.35
    if 11 <= hour <= 14:
        return 1.10
    return 1.0


def simulate_stage_duration(stage: str, weather_mult: float, tod_mult: float) -> float:
    p = STAGE_PARAMS[stage]
    sampled = np.random.gamma(shape=p["shape"], scale=p["scale"])
    # transit and pickup affected by weather & time-of-day
    if stage in ("pickup_to_transit", "transit_to_delivered"):
        sampled *= weather_mult * tod_mult
    return round(max(sampled, 0.1), 2)


def simulate_ch_features() -> dict:
    """Simulate graph engine (P1) derived features for one order."""
    ch_distance_km = round(np.random.gamma(shape=3.0, scale=2.0), 2)   # km
    segment_count  = int(np.random.poisson(lam=18))
    turn_count     = int(np.random.poisson(lam=max(1, segment_count // 3)))
    road_dist      = np.random.dirichlet(alpha=[2, 4, 3, 2])            # fractions
    road_type_dist = {rt: round(float(road_dist[i]), 3)
                      for i, rt in enumerate(ROAD_TYPES)}
    return {
        "ch_distance_km":           ch_distance_km,
        "segment_count":            segment_count,
        "turn_count":               turn_count,
        "road_highway_frac":        road_type_dist["highway"],
        "road_primary_frac":        road_type_dist["primary"],
        "road_secondary_frac":      road_type_dist["secondary"],
        "road_residential_frac":    road_type_dist["residential"],
    }


def simulate_dag_features(stage_durations: dict) -> dict:
    """
    Simulate DAG (P2) derived features.
    cpm_slack – slack from CPM run (minutes the project could slip before ETA breach).
    """
    total_eta   = sum(stage_durations.values())
    cpm_slack   = round(np.random.exponential(scale=2.5), 2)
    batch_size  = int(np.random.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1]))
    prep_var    = round(float(np.var(
        np.random.gamma(shape=4.0, scale=2.0, size=50)
    )), 4)
    return {
        "cpm_slack_min":          cpm_slack,
        "batch_size":             batch_size,
        "estimated_prep_variance":prep_var,
        "stage_delay_placed":     stage_durations["placed_to_accepted"],
        "stage_delay_prep":       stage_durations["accepted_to_prep"],
        "stage_delay_pickup":     stage_durations["prep_to_pickup"],
        "stage_delay_transit":    stage_durations["pickup_to_transit"],
        "stage_delay_delivered":  stage_durations["transit_to_delivered"],
    }


# ── main simulation ───────────────────────────────────────────────────────────

def simulate_orders(n: int = N_ORDERS) -> pd.DataFrame:
    records = []

    for order_id in range(1, n + 1):
        # ── temporal features ──────────────────────────────────────────────
        ts          = random_timestamp(START_DATE, END_DATE)
        hour        = ts.hour
        dow         = ts.weekday()          # 0=Mon … 6=Sun
        month       = ts.month
        is_weekend  = int(dow >= 5)

        # ── weather ────────────────────────────────────────────────────────
        weather     = np.random.choice(list(WEATHER_PROFILES.keys()),
                                        p=WEATHER_PROBS)
        weather_mult = WEATHER_PROFILES[weather]
        tod_mult     = time_of_day_multiplier(hour)

        # ── stage durations ────────────────────────────────────────────────
        stage_durations = {
            stage: simulate_stage_duration(stage, weather_mult, tod_mult)
            for stage in STAGE_PARAMS
        }
        ground_truth_eta = round(sum(stage_durations.values()), 2)

        # ── graph (P1) features ────────────────────────────────────────────
        ch_feats  = simulate_ch_features()

        # ── DAG (P2) features ──────────────────────────────────────────────
        dag_feats = simulate_dag_features(stage_durations)

        # ── late flag ──────────────────────────────────────────────────────
        promised_eta = 30.0   # platform SLA in minutes
        is_late      = int(ground_truth_eta > promised_eta)

        record = {
            "order_id":         order_id,
            "timestamp":        ts,
            "hour_of_day":      hour,
            "day_of_week":      dow,
            "month":            month,
            "is_weekend":       is_weekend,
            "weather":          weather,
            "weather_mult":     weather_mult,
            "tod_mult":         round(tod_mult, 2),
            "ground_truth_eta": ground_truth_eta,
            "is_late":          is_late,
            **ch_feats,
            **dag_feats,
        }
        records.append(record)

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    return df


def split_by_date(df: pd.DataFrame,
                  train_end: str = "2024-09-30",
                  val_end:   str = "2024-10-31") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split – avoids leakage."""
    train = df[df["timestamp"] <= train_end]
    val   = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)]
    test  = df[df["timestamp"] > val_end]
    return train, val, test


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Simulating delivery orders …")
    df = simulate_orders(N_ORDERS)
    print(f"Total orders simulated : {len(df)}")
    print(f"Date range             : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"Late delivery rate     : {df['is_late'].mean():.1%}")

    train, val, test = split_by_date(df)
    print(f"\nSplit sizes  →  train={len(train)}  val={len(val)}  test={len(test)}")

    df.to_csv(   os.path.join(OUTPUT_DIR, "all_orders.csv"),   index=False)
    train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"),        index=False)
    val.to_csv(  os.path.join(OUTPUT_DIR, "val.csv"),          index=False)
    test.to_csv( os.path.join(OUTPUT_DIR, "test.csv"),         index=False)

    print("\nFiles saved to Data/")
    print("  all_orders.csv")
    print("  train.csv")
    print("  val.csv")
    print("  test.csv")
    print("\nColumn list:")
    for col in df.columns:
        print(f"  {col}")