import numpy as np
from dag_builder import build_dag


def sample_durations(edge_data, seed=None):
    if seed is not None:
        np.random.seed(seed)

    sampled = {}
    for (u, v), (base_dur, shape, scale) in edge_data.items():
        sampled[(u, v)] = round(np.random.gamma(shape=shape, scale=scale), 2)

    prep_samples = np.random.gamma(shape=4.0, scale=2.0, size=1000)
    prep_variance = round(float(np.var(prep_samples)), 4)

    return sampled, prep_variance


if __name__ == "__main__":
    _, _, edge_data = build_dag()
    sampled, prep_var = sample_durations(edge_data, seed=42)
    print("Sampled durations:")
    for (u, v), dur in sampled.items():
        print(f"  {u} -> {v}: {dur} min")
    print(f"Prep variance (1000 samples): {prep_var} min²")