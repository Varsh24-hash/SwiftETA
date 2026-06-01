from dag_builder import build_dag, STAGES


def kahns_sort(adj, in_degree):
    in_deg = dict(in_degree)
    queue = [n for n in STAGES if in_deg[n] == 0]
    topo_order = []

    while queue:
        node = queue.pop(0)
        topo_order.append(node)

        for (successor, _, _, _) in adj[node]:
            in_deg[successor] -= 1
            if in_deg[successor] == 0:
                queue.append(successor)

    if len(topo_order) != len(STAGES):
        raise ValueError("Cycle detected in DAG!")

    return topo_order


if __name__ == "__main__":
    adj, in_degree, _ = build_dag()
    order = kahns_sort(adj, in_degree)
    print("Topological order:", order)