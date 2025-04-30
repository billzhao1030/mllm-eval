import os
import json
import math

import numpy as np
import networkx as nx
from tqdm import tqdm

import plotly.graph_objects as go

from huggingface_hub import hf_hub_download


def load_hub_data(repo_id, filename, extension="json"):
    json_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset"
    )

    if extension.lower() == "json":
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    elif extension.lower() == "txt":
        with open(json_path, 'r') as f:
            data = [line.strip() for line in f]
        return data
    else:
        print(f"Unsupported extension: {extension}.")
        return None

def load_graph(location_data, navigable_data, scans):
    """
    Load graph from scan,
    Store the graph {scan_id: graph} in graphs
    Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in paths
    Store the distances in distances. (Structure see above)
    Load connectivity graph for each scan, useful for reasoning about shortest paths
    """
    
    graphs = load_nav_graphs(location_data, navigable_data, scans)

    shortest_paths = {}
    graphs_bar = tqdm(
        graphs.items(),
        desc='Computing shortest paths',
    )
    for scan, G in graphs_bar:  # compute all shortest paths
        shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))

    shortest_distances = {}
    
    graphs_bar = tqdm(
        graphs.items(),
        desc='Computing shortest distances',
    )
    for scan, G in graphs_bar:  # compute all shortest paths
        shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
    
    return graphs, shortest_paths, shortest_distances


def load_nav_graphs(location_data, navigable_data, scans):
    ''' Load connectivity graphs from pre-extracted location and navigable JSON files '''

    graphs = {}

    scans = tqdm(
        scans,
        desc='Loading navigation graphs',
    )

    for scan in scans:
        G = nx.Graph()
        positions = location_data[scan]
        navigables = navigable_data[scan]

        # Add nodes with positions
        for viewpoint_id, coords in positions.items():
            G.add_node(viewpoint_id, position=np.array(coords))

        # Add edges using navigability info
        for src_viewpoint, neighbors in navigables.items():
            for tgt_viewpoint, nav_info in neighbors.items():
                distance = nav_info['distance']
                G.add_edge(src_viewpoint, tgt_viewpoint, weight=distance)

        graphs[scan] = G

    return graphs

def load_nav_graphs_from_connectivity(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3] - pose2['pose'][3]) ** 2 \
                + (pose1['pose'][7] - pose2['pose'][7]) ** 2 \
                + (pose1['pose'][11] - pose2['pose'][11]) ** 2) ** 0.5

    graphs = {}
    scans = tqdm(
        scans,
        desc='Loading navigation graphs from connectivity files',
    )

    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item['included']:
                    for j, conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'], data[j]['image_id'], weight=distance(item, data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

def visualize_graph_3d_rotating(G, title="3D Navigation Graph"):
    pos = nx.get_node_attributes(G, 'position')
    edge_x = []
    edge_y = []
    edge_z = []
    for (u, v, d) in G.edges(data=True):
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_z = []
    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(size=5, color='skyblue'),
        text=list(G.nodes()), # Add node IDs for hover information
        hoverinfo='text'
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    # Define the rotation animation
    frames = []
    num_steps = 150  # Number of frames in the rotation
    for i in range(num_steps):
        angle = (i / num_steps) * 360
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),  # Adjust if your graph is centered elsewhere
            eye=dict(x=2.0 * np.cos(np.radians(angle)), y=2.0 * np.sin(np.radians(angle)), z=1.0) # Adjust zoom (2.0) and vertical position (1.0)
        )
        frames.append(go.Frame(layout=dict(scene_camera=camera)))

    fig.update(frames=frames)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data' # Ensure proper scaling of the 3D space
        ),
        updatemenus=[{
            "type": "buttons",
            "buttons": [{
                "args": [None, {"frame": {"duration": 30, "redraw": True},
                                "fromcurrent": True, "transition": {"duration": 30,
                                                                    "easing": "linear"}}],
                "label": "Play",
                "method": "animate"
            }, {
                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
                "label": "Pause",
                "method": "animate"
            }]
        }]
    )
    fig.show()