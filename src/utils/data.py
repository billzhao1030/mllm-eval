import os
import json
import pickle

import numpy as np
import networkx as nx
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, Optional, NamedTuple, Union

import plotly.graph_objects as go

from datasets import load_dataset
from huggingface_hub import hf_hub_download

def load_hub_data(repo_id, filename, extension="json", save_dir="../data"):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    file_path = f"{save_dir}/{filename}"

    if not os.path.exists(file_path):
        print(f"{file_path} not exist, downloading from Hugging Face...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=save_dir
        )
    else:
        print(f"{file_path} exist, loading data...")

    if extension.lower() == "json":
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    elif extension.lower() == "txt":
        with open(file_path, 'r') as f:
            data = [line.strip() for line in f]
        return data
    elif extension.lower() == "pkl":
        with open(file_path, "rb") as f:
            data = pickle.load(f)
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

    shortest_paths = load_hub_data("billzhao1030/MP3D", "graphs/shortest_paths.pkl", extension="pkl")
    shortest_distances = load_hub_data("billzhao1030/MP3D", "graphs/shortest_distances.pkl", extension="pkl")
    
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

def download_mp3d_observations(logger, save_dir="../data/observations", overwrite=False):
    """
    Download MP3D feature observations scan-by-scan from Hugging Face.
    
    Parameters:
        save_dir (str): Directory to save the observations.
        overwrite (bool): If True, re-download even if scan already exists.
    """
    # Check if the save_dir exists
    if os.path.exists(save_dir) and not overwrite:
        logger.info(f"🟢 Observation folder '{save_dir}' already exists. Skipping download.")
        return

    # Ensure save_dir is created if not exists
    os.makedirs(save_dir, exist_ok=True)

    # Load scan IDs
    scan_ids = load_hub_data("billzhao1030/MP3D", "scans.txt", extension="txt")
    dataset_repo = "billzhao1030/MP3D_feature"

    # Download scan-by-scan
    for scan_id in tqdm(scan_ids, desc="Processing scans"):
        scan_folder = os.path.join(save_dir, scan_id)
        os.makedirs(scan_folder, exist_ok=True)

        if not overwrite and os.listdir(scan_folder):
            logger.info(f"Scan {scan_id} already downloaded, skipping.")
            continue

        logger.info(f"\n⬇️  Downloading split (scan): {scan_id}")
        scan_dataset = load_dataset(dataset_repo, split=scan_id, streaming=True)

        for item in tqdm(scan_dataset, desc=f"Downloading {scan_id}", leave=False):
            viewpoint_id = item["viewpoint_id"]

            for i in range(4):
                img = item[f"image_{i}"]
                img_save_path = os.path.join(scan_folder, f"{viewpoint_id}_{i}.png")
                img.save(img_save_path)

        logger.info(f"✅ Finished scan {scan_id}.\n")

    logger.info("🏁 All scans finished.")

@dataclass
class AgentAction:
    """A full description of an action for an ActionAgent to execute."""

    tool: str
    """The name of the Tool to execute."""
    tool_input: Union[str, dict]
    """The input to pass in to the Tool."""
    log: str
    """Additional information to log about the action."""


class AgentFinish(NamedTuple):
    """The final return value of an ActionAgent."""

    return_values: dict
    """Dictionary of return values."""
    log: str
    """Additional information to log about the return value"""

class OutputParserException(ValueError):
    """Exception that output parsers should raise to signify a parsing error.

    This exists to differentiate parsing errors from other code or execution errors
    that also may arise inside the output parser. OutputParserExceptions will be
    available to catch and handle in ways to fix the parsing error, while other
    errors will be raised.

    Args:
        error: The error that's being re-raised or an error message.
        observation: String explanation of error which can be passed to a
            model to try and remediate the issue.
        llm_output: String model output which is error-ing.
        send_to_llm: Whether to send the observation and llm_output back to an Agent
            after an OutputParserException has been raised. This gives the underlying
            model driving the agent the context that the previous output was improperly
            structured, in the hopes that it will update the output to the correct
            format.
    """

    def __init__(
        self,
        error: Any,
        observation: Optional[str] = None,
        llm_output: Optional[str] = None,
        send_to_llm: bool = False,
    ):
        super(OutputParserException, self).__init__(error)
        if send_to_llm:
            if observation is None or llm_output is None:
                raise ValueError(
                    "Arguments 'observation' & 'llm_output'"
                    " are required if 'send_to_llm' is True"
                )
        self.observation = observation
        self.llm_output = llm_output
        self.send_to_llm = send_to_llm