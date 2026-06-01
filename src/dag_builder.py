STAGES = [
    "placed",
    "accepted",
    "prep",
    "pickup",
    "transit",
    "delivered"
]

EDGES = [
    ("placed",   "accepted",  1.5,  3.0, 0.5),
    ("accepted", "prep",      8.0,  4.0, 2.0),
    ("prep",     "pickup",    3.0,  3.0, 1.0),
    ("pickup",   "transit",  12.0,  6.0, 2.0),
    ("transit",  "delivered", 0.5,  2.0, 0.25),
]


def build_dag():
    adj = {s: [] for s in STAGES}
    in_degree = {s: 0 for s in STAGES}
    edge_data = {}

    for (u, v, dur, shape, scale) in EDGES:
        adj[u].append((v, dur, shape, scale))
        in_degree[v] += 1
        edge_data[(u, v)] = (dur, shape, scale)

    return adj, in_degree, edge_data


if __name__ == "__main__":
    adj, in_degree, edge_data = build_dag()
    print("Stages:", STAGES)
    print("In-degrees:", in_degree)
    print("Edges:")
    for (u, v), (dur, shape, scale) in edge_data.items():
        print(f"  {u} -> {v}  base={dur}min  Gamma(shape={shape}, scale={scale})")