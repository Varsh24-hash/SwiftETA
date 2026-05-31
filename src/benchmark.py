import osmnx as ox
import networkx as nx
import random
import time

print("Loading graph...")

G = ox.load_graphml("Data/bengaluru.graphml")

nodes = list(G.nodes)

query_count = 100
total_time = 0

print(f"Running {query_count} shortest path queries...")

for i in range(query_count):

    source = random.choice(nodes)
    target = random.choice(nodes)

    start = time.time()

    try:
        nx.shortest_path(
            G,
            source,
            target,
            weight="length"
        )

    except:
        continue

    end = time.time()

    total_time += (end - start)

avg_time = total_time / query_count

print("\nBenchmark Results")
print("Queries Executed:", query_count)
print("Average Dijkstra Time:", avg_time, "seconds")