import os
import pickle

import networkx as nx
from tqdm import tqdm
from utils.data import load_hub_data, load_nav_graphs

cache_dir='../data/graph_cache'

os.makedirs(cache_dir, exist_ok=True)

paths_path = os.path.join(cache_dir, 'shortest_paths.pkl')
dists_path = os.path.join(cache_dir, 'shortest_distances.pkl')

navigable_data = load_hub_data("billzhao1030/MP3D", "navigable.json")
location_data = load_hub_data("billzhao1030/MP3D", "location.json")
scans = load_hub_data("billzhao1030/MP3D", "scans.txt", extension="txt")

graphs = load_nav_graphs(location_data, navigable_data, scans)

# Calculate the shortest path and distance dict
shortest_paths = {}
graphs_bar = tqdm(graphs.items(), desc='Computing shortest paths')
for scan, G in graphs_bar:
    shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))

shortest_distances = {}
graphs_bar = tqdm(graphs.items(), desc='Computing shortest distances')
for scan, G in graphs_bar:
    shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

print("Save shortest path")
with open(paths_path, 'wb') as f:
    pickle.dump(shortest_paths, f)

print("Save shortest distance")
with open(dists_path, 'wb') as f:
    pickle.dump(shortest_distances, f)


from huggingface_hub import upload_file

repo_id = "billzhao1030/MP3D"  # or your actual repo
repo_type = "dataset"

# Upload the shortest_paths file
upload_file(
    path_or_fileobj=f"{cache_dir}/shortest_paths.pkl",
    path_in_repo="shortest_paths.pkl",  # upload under a folder if you like
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Shortest path file uploaded!")

# Upload the shortest_distances file
upload_file(
    path_or_fileobj=f"{cache_dir}/shortest_distances.pkl",
    path_in_repo="shortest_distances.pkl",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Shortest distance file uploaded!")