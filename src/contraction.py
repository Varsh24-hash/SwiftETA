import osmnx as ox

print("Loading graph...")

# Load Bengaluru graph
G = ox.load_graphml("Data/bengaluru.graphml")

print("Graph loaded!")
print("Nodes:", len(G.nodes))
print("Edges:", len(G.edges))

original_edges = len(G.edges)

# -----------------------------
# STEP 1: Calculate Importance
# -----------------------------

print("\nCalculating node importance...")

importance = {}

for node in G.nodes:

    importance[node] = (
        G.degree(node)
        + len(list(G.predecessors(node)))
        + len(list(G.successors(node)))
    )

print("Importance calculated!")

# -----------------------------
# STEP 2: Sort Nodes
# -----------------------------

sorted_nodes = sorted(
    importance,
    key=importance.get
)

print("\nLowest Importance Nodes:")

for node in sorted_nodes[:10]:
    print(node, "Importance =", importance[node])

print("\nRanking completed!")

# -----------------------------
# STEP 3: Create Shortcut Edges
# -----------------------------

print("\nCreating shortcut edges...")

shortcut_count = 0

for node in sorted_nodes:

    if G.degree(node) < 2:
        continue

    try:
        preds = list(G.predecessors(node))
        succs = list(G.successors(node))

        for u in preds:
            for v in succs:

                if u == v:
                    continue

                if not G.has_edge(u, v):

                    try:

                        edge1 = G.get_edge_data(u, node)
                        edge2 = G.get_edge_data(node, v)

                        w1 = float(
                            list(edge1.values())[0]["length"]
                        )

                        w2 = float(
                            list(edge2.values())[0]["length"]
                        )

                        shortcut_length = w1 + w2

                        G.add_edge(
                            u,
                            v,
                            length=shortcut_length
                        )

                        shortcut_count += 1

                    except:
                        continue

        # Limit for demo/project
        if shortcut_count >= 100:
            break

    except:
        continue

print("\nShortcut edges added:", shortcut_count)

# -----------------------------
# STEP 4: Save Contracted Graph
# -----------------------------

print("\nSaving contracted graph...")

ox.save_graphml(
    G,
    "Data/contracted_bengaluru.graphml"
)

print("Contracted graph saved!")

# -----------------------------
# STEP 5: Final Statistics
# -----------------------------

print("\nFinal Graph Statistics")

print("Nodes:", len(G.nodes))
print("Original Edges:", original_edges)
print("Current Edges:", len(G.edges))
print("Shortcut Edges Added:", shortcut_count)

print("\nContraction Hierarchy preprocessing completed!")