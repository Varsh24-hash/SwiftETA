import osmnx as ox
import os

print("Downloading Bengaluru road network...")

G = ox.graph_from_place(
    "Bengaluru, Karnataka, India",
    network_type="drive"
)

print("Number of Nodes:", len(G.nodes))
print("Number of Edges:", len(G.edges))

# Absolute path
save_path = os.path.join(os.getcwd(), "Data", "bengaluru.graphml")

print("Saving to:", save_path)

ox.save_graphml(G, save_path)

print("Graph saved successfully!")