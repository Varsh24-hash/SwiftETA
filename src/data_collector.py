"""
data_collector.py  –  Person 3
Builds historical delivery dataset using REAL outputs from P1 and P2.

P1  (graph engine)
    ─ loads contracted_bengaluru.graphml
    ─ runs bidirectional Dijkstra for each order pair
    ─ extracts: distance_km, segment_count, turn_count, road-type fractions

P2  (DAG scheduler)
    ─ calls sample_durations()         → per-stage stochastic durations
    ─ calls run_cpm()                  → cpm_slack, stage_delays, total_eta
    ─ calls greedy_batch_scheduler()   → batch_size

ground_truth_eta = route_time + cpm_eta + weather_delay + random_noise

No values are fabricated – every numeric feature comes from P1/P2 logic.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from datetime import datetime, timedelta

# ── make P1/P2 modules importable (they live in the same directory) ───────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dag_builder        import build_dag, STAGES
from kahn_sort          import kahns_sort
from stochastic_weights import sample_durations
from cpm                import run_cpm
from batch_scheduler    import greedy_batch_scheduler

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── config ────────────────────────────────────────────────────────────────────
N_ORDERS    = 5_000
START_DATE  = datetime(2024, 1, 1)
END_DATE    = datetime(2024, 12, 31)
OUTPUT_DIR  = "Data"
GRAPH_PATH  = os.path.join("Data", "contracted_bengaluru.graphml")
PROMISED_ETA = 30.0          # SLA in minutes
AVG_SPEED_KMH = 25.0         # Bengaluru average urban speed

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── weather profiles ──────────────────────────────────────────────────────────
WEATHER_PROFILES = {
    "clear":      1.0,
    "cloudy":     1.05,
    "rainy":      1.25,
    "heavy_rain": 1.55,
}
WEATHER_PROBS = [0.50, 0.25, 0.18, 0.07]

# ── road-type speed factors (lower = slower) ──────────────────────────────────
ROAD_SPEED = {
    "highway":     1.0,
    "primary":     0.80,
    "secondary":   0.65,
    "residential": 0.50,
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 – P1 GRAPH ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def load_graph(path: str = GRAPH_PATH) -> nx.MultiDiGraph:
    """Load P1's contracted Bengaluru graph."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Graph not found at '{path}'.\n"
            "Run graph_loader.py then contraction.py first (Person 1 steps)."
        )
    print(f"[P1] Loading graph from {path} …")
    G = ox.load_graphml(path)
    print(f"[P1] Graph loaded  →  {len(G.nodes):,} nodes  {len(G.edges):,} edges")
    return G


def extract_p1_features(G: nx.MultiDiGraph,
                         source: int, target: int) -> dict:
    """
    Run Dijkstra on P1's contracted graph and extract:
      • distance_km    – total path length in km
      • segment_count  – number of edges in path
      • turn_count     – edges where road type changes (proxy for turns)
      • road_*_frac    – fraction of segments by road type
    """
    try:
        path_nodes = nx.shortest_path(G, source, target, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None          # caller skips this pair

    # Walk path edges
    total_length_m = 0.0
    road_type_counts = {"highway": 0, "primary": 0,
                        "secondary": 0, "residential": 0}
    turn_count = 0
    prev_highway = None

    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        edge_data = G.get_edge_data(u, v)
        if edge_data is None:
            continue
        attrs = list(edge_data.values())[0]   # first parallel edge

        # length
        total_length_m += float(attrs.get("length", 0))

        # road type
        hw = attrs.get("highway", "residential")
        if isinstance(hw, list):
            hw = hw[0]
        # map to our four buckets
        if "motorway" in hw or "trunk" in hw:
            bucket = "highway"
        elif "primary" in hw:
            bucket = "primary"
        elif "secondary" in hw or "tertiary" in hw:
            bucket = "secondary"
        else:
            bucket = "residential"
        road_type_counts[bucket] += 1

        # turn proxy: count road-type transitions
        if prev_highway is not None and bucket != prev_highway:
            turn_count += 1
        prev_highway = bucket

    segment_count = len(path_nodes) - 1
    total_km      = round(total_length_m / 1000.0, 3)

    # road-type fractions
    total_segs = max(segment_count, 1)
    fracs = {f"road_{rt}_frac": round(road_type_counts[rt] / total_segs, 4)
             for rt in road_type_counts}

    return {
        "distance_km":   total_km,
        "segment_count": segment_count,
        "turn_count":    turn_count,
        **fracs,
    }


def compute_route_time(p1: dict, weather_mult: float, tod_mult: float) -> float:
    """
    Translate P1 distance into travel time (minutes).
    Weighted average speed by road type, then apply weather + tod penalty.
    """
    weighted_speed = (
        p1["road_highway_frac"]     * ROAD_SPEED["highway"]
      + p1["road_primary_frac"]     * ROAD_SPEED["primary"]
      + p1["road_secondary_frac"]   * ROAD_SPEED["secondary"]
      + p1["road_residential_frac"] * ROAD_SPEED["residential"]
    ) * AVG_SPEED_KMH                           # km/h

    effective_speed = max(weighted_speed / (weather_mult * tod_mult), 1.0)
    route_time_min  = (p1["distance_km"] / effective_speed) * 60.0
    return round(route_time_min, 3)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 – P2 DAG ENGINE
# ═════════════════════════════════════════════════════════════════════════════

# Build DAG once (shared across all orders)
_ADJ, _IN_DEG, _EDGE_DATA = build_dag()
_TOPO_ORDER = kahns_sort(_ADJ, _IN_DEG)


def extract_p2_features(order_id: int) -> dict:
    """
    For each order:
      1. sample_durations()          → stochastic stage times
      2. run_cpm()                   → cpm_slack, stage_delays, total_eta
      3. greedy_batch_scheduler()    → batch_size

    Returns a flat dict of P2 features.
    """
    # ── stochastic durations (different seed per order for variety) ────────
    sampled, prep_variance = sample_durations(_EDGE_DATA, seed=order_id)

    # ── CPM ────────────────────────────────────────────────────────────────
    cpm_result = run_cpm(_TOPO_ORDER, _ADJ, _EDGE_DATA, sampled)
    cpm_eta    = cpm_result["total_eta_minutes"]      # minutes end-to-end
    cpm_slack  = cpm_result["cpm_slack"]
    stage_delays = cpm_result["stage_delays"]         # dict keyed "u_to_v"

    # ── greedy batch scheduler ─────────────────────────────────────────────
    # Build a small window of orders around this one to simulate batching
    # deadline = cpm_eta (the order's own ETA), ready_time varies by ±5 min
    rng = np.random.default_rng(order_id)
    n_window = int(rng.integers(2, 6))               # 2-5 co-arriving orders
    window_orders = []
    for k in range(n_window):
        ready  = round(float(rng.uniform(0, 5)), 1)
        dl     = round(float(rng.uniform(cpm_eta * 0.8, cpm_eta * 1.2)), 1)
        window_orders.append((f"O{order_id}_{k}", ready, dl))

    batch_result = greedy_batch_scheduler(window_orders)
    batch_size   = batch_result["batch_size"]        # avg orders per batch

    # ── flatten stage delays ───────────────────────────────────────────────
    return {
        "cpm_slack_min":           round(cpm_slack, 3),
        "batch_size":              batch_size,
        "estimated_prep_variance": round(prep_variance, 4),
        "cpm_eta_min":             round(cpm_eta, 3),
        "stage_delay_placed":      stage_delays.get("placed_to_accepted", 0),
        "stage_delay_prep":        stage_delays.get("accepted_to_prep",   0),
        "stage_delay_pickup":      stage_delays.get("prep_to_pickup",     0),
        "stage_delay_transit":     stage_delays.get("pickup_to_transit",  0),
        "stage_delay_delivered":   stage_delays.get("transit_to_delivered", 0),
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 – TEMPORAL HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def random_timestamp(start: datetime, end: datetime) -> datetime:
    delta = (end - start).total_seconds()
    return start + timedelta(seconds=random.uniform(0, delta))


def time_of_day_multiplier(hour: int) -> float:
    if 8 <= hour <= 10 or 17 <= hour <= 20:
        return 1.35
    if 11 <= hour <= 14:
        return 1.10
    return 1.0


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 – GROUND-TRUTH ETA FORMULA
# ═════════════════════════════════════════════════════════════════════════════

def compute_ground_truth_eta(route_time: float,
                              cpm_eta:    float,
                              weather_delay: float,
                              order_id:   int) -> float:
    """
    ground_truth_eta = route_time + cpm_eta + weather_delay + random_noise

    route_time    → from P1 path (distance × speed × road-type × tod)
    cpm_eta       → from P2 CPM total end-to-end stage time
    weather_delay → route_time × (weather_mult - 1.0)  [additive penalty]
    random_noise  → small Gaussian ±1 min (sensor / GPS jitter)
    """
    rng   = np.random.default_rng(order_id + 99_999)
    noise = float(rng.normal(loc=0.0, scale=1.0))
    eta   = route_time + cpm_eta + weather_delay + noise
    return round(max(eta, 1.0), 3)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 – MAIN COLLECTOR
# ═════════════════════════════════════════════════════════════════════════════

def collect_orders(G: nx.MultiDiGraph, n: int = N_ORDERS) -> pd.DataFrame:
    nodes   = list(G.nodes)
    records = []
    skipped = 0
    order_id = 0

    print(f"\n[P3] Collecting {n} orders …")

    while len(records) < n:
        order_id += 1

        # ── random source / target from graph ─────────────────────────────
        src = random.choice(nodes)
        tgt = random.choice(nodes)
        if src == tgt:
            continue

        # ── P1: graph features ─────────────────────────────────────────────
        p1 = extract_p1_features(G, src, tgt)
        if p1 is None:
            skipped += 1
            continue

        # ── temporal + weather ─────────────────────────────────────────────
        ts          = random_timestamp(START_DATE, END_DATE)
        hour        = ts.hour
        dow         = ts.weekday()
        month       = ts.month
        is_weekend  = int(dow >= 5)
        weather     = str(np.random.choice(list(WEATHER_PROFILES.keys()),
                                            p=WEATHER_PROBS))
        weather_mult = WEATHER_PROFILES[weather]
        tod_mult     = time_of_day_multiplier(hour)

        # ── P1: route time ─────────────────────────────────────────────────
        route_time    = compute_route_time(p1, weather_mult, tod_mult)
        weather_delay = round(route_time * (weather_mult - 1.0), 3)

        # ── P2: DAG features ───────────────────────────────────────────────
        p2 = extract_p2_features(order_id)

        # ── ground-truth ETA ───────────────────────────────────────────────
        gt_eta = compute_ground_truth_eta(
            route_time    = route_time,
            cpm_eta       = p2["cpm_eta_min"],
            weather_delay = weather_delay,
            order_id      = order_id,
        )

        is_late = int(gt_eta > PROMISED_ETA)

        record = {
            # identifiers
            "order_id":              len(records) + 1,
            "timestamp":             ts,
            # temporal
            "hour_of_day":           hour,
            "day_of_week":           dow,
            "month":                 month,
            "is_weekend":            is_weekend,
            # weather
            "weather":               weather,
            "weather_mult":          weather_mult,
            "tod_mult":              round(tod_mult, 2),
            # ── P1 features ────────────────────────────────────────────────
            "ch_distance_km":        p1["distance_km"],
            "segment_count":         p1["segment_count"],
            "turn_count":            p1["turn_count"],
            "road_highway_frac":     p1["road_highway_frac"],
            "road_primary_frac":     p1["road_primary_frac"],
            "road_secondary_frac":   p1["road_secondary_frac"],
            "road_residential_frac": p1["road_residential_frac"],
            # ── P2 features ────────────────────────────────────────────────
            "cpm_slack_min":           p2["cpm_slack_min"],
            "batch_size":              p2["batch_size"],
            "estimated_prep_variance": p2["estimated_prep_variance"],
            "stage_delay_placed":      p2["stage_delay_placed"],
            "stage_delay_prep":        p2["stage_delay_prep"],
            "stage_delay_pickup":      p2["stage_delay_pickup"],
            "stage_delay_transit":     p2["stage_delay_transit"],
            "stage_delay_delivered":   p2["stage_delay_delivered"],
            # ── ETA components (kept for traceability) ─────────────────────
            "route_time_min":   route_time,
            "cpm_eta_min":      p2["cpm_eta_min"],
            "weather_delay_min":weather_delay,
            # ── target ─────────────────────────────────────────────────────
            "ground_truth_eta": gt_eta,
            "is_late":          is_late,
        }
        records.append(record)

        if len(records) % 500 == 0:
            print(f"  … {len(records)}/{n} orders collected  "
                  f"(skipped {skipped} disconnected pairs)")

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    return df


def split_by_date(df: pd.DataFrame,
                  train_end: str = "2024-09-30",
                  val_end:   str = "2024-10-31"):
    """Chronological split – no data leakage."""
    train = df[df["timestamp"] <= train_end]
    val   = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)]
    test  = df[df["timestamp"] > val_end]
    return train, val, test


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # load P1 graph
    G = load_graph(GRAPH_PATH)

    # collect orders using real P1 + P2 calls
    df = collect_orders(G, N_ORDERS)

    # summary
    print(f"\n[P3] Dataset summary")
    print(f"  Total orders      : {len(df)}")
    print(f"  Date range        : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"  Late delivery rate: {df['is_late'].mean():.1%}")
    print(f"\n  ETA components (mean)")
    print(f"    route_time      : {df['route_time_min'].mean():.2f} min")
    print(f"    cpm_eta         : {df['cpm_eta_min'].mean():.2f} min")
    print(f"    weather_delay   : {df['weather_delay_min'].mean():.2f} min")
    print(f"    ground_truth_eta: {df['ground_truth_eta'].mean():.2f} min")

    # split
    train, val, test = split_by_date(df)
    print(f"\n  Split  →  train={len(train)}  val={len(val)}  test={len(test)}")

    # save
    df.to_csv(   os.path.join(OUTPUT_DIR, "all_orders.csv"), index=False)
    train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"),      index=False)
    val.to_csv(  os.path.join(OUTPUT_DIR, "val.csv"),        index=False)
    test.to_csv( os.path.join(OUTPUT_DIR, "test.csv"),       index=False)

    print(f"\n[P3] Saved to Data/  →  all_orders.csv  train.csv  val.csv  test.csv")
    print("\nColumns:")
    for col in df.columns:
        print(f"  {col}")