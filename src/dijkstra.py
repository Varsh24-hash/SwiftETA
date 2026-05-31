import osmnx as ox
import networkx as nx
import random
import time

print("Loading graph...")

G = ox.load_graphml("Data/bengaluru.graphml")

print("Graph loaded!")

nodes = list(G.nodes)

source = random.choice(nodes)
target = random.choice(nodes)

print("Source:", source)
print("Target:", target)

start = time.time()

path = nx.shortest_path(
    G,
    source,
    target,
    weight="length"
)

end = time.time()

print("\nPath Found!")
print("Nodes in Path:", len(path))
print("Execution Time:", end-start, "seconds")