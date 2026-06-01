from dag_builder import build_dag, STAGES
from kahn_sort import kahns_sort
from stochastic_weights import sample_durations


def run_cpm(topo_order, adj, edge_data, sampled_durations):
    earliest_finish = {s: 0.0 for s in STAGES}

    for stage in topo_order:
        for (successor, _, _, _) in adj[stage]:
            duration = sampled_durations.get((stage, successor), 0)
            candidate = earliest_finish[stage] + duration
            if candidate > earliest_finish[successor]:
                earliest_finish[successor] = round(candidate, 2)

    project_duration = earliest_finish["delivered"]

    latest_finish = {s: 0.0 for s in STAGES}
    latest_finish["delivered"] = project_duration

    for stage in reversed(topo_order):
        successors = [v for (v, _, _, _) in adj[stage]]
        if not successors:
            latest_finish[stage] = earliest_finish[stage]
        else:
            candidates = []
            for succ in successors:
                dur = sampled_durations.get((stage, succ), 0)
                candidates.append(latest_finish[succ] - dur)
            latest_finish[stage] = round(min(candidates), 2)

    slack = {}
    critical_path = []
    cpm_slack = 0.0

    for stage in topo_order:
        s = round(latest_finish[stage] - earliest_finish[stage], 2)
        slack[stage] = s
        if s == 0.0:
            critical_path.append(stage)
        else:
            cpm_slack += s

    stage_delays = {
        f"{u}_to_{v}": dur for (u, v), dur in sampled_durations.items()
    }

    return {
        "earliest_finish": earliest_finish,
        "latest_finish": latest_finish,
        "slack": slack,
        "critical_path": critical_path,
        "cpm_slack": round(cpm_slack, 2),
        "stage_delays": stage_delays,
        "total_eta_minutes": project_duration
    }


if __name__ == "__main__":
    adj, in_degree, edge_data = build_dag()
    topo_order = kahns_sort(adj, in_degree)
    sampled, _ = sample_durations(edge_data, seed=42)
    result = run_cpm(topo_order, adj, edge_data, sampled)

    print("Critical path:", " -> ".join(result["critical_path"]))
    print("Total ETA:", result["total_eta_minutes"], "min")
    print("Slack per stage:")
    for stage, s in result["slack"].items():
        print(f"  {stage}: {s} min")