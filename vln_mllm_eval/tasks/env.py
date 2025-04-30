import math
import numpy as np
import random
import networkx as nx
from collections import defaultdict

from utils.data import load_nav_graphs, load_hub_data
from utils.eval import cal_dtw, cal_cls
from utils.graph import NavGraph

ERROR_MARGIN = 3.0

class Simulator(object):
    """A simple simulator in Matterport3D environment"""

    def __init__(self, env_data):
        self.heading = 0
        self.elevation = 0
        self.scan_ID = ''
        self.viewpoint_ID = ''
        
        self.env_navigable = env_data["navigable"]
        self.env_loaction = env_data["location"]
        self.location = None
        self.navigable = {}
        self.candidate = {}
        self.gmap = NavGraph()

    def _make_id(self, scan_ID, viewpoint_ID):
        return scan_ID + '_' + viewpoint_ID

    def newEpisode(
        self, 
        scan_ID: str, 
        viewpoint_ID: str,
        heading: int, 
        elevation: int
    ):
        """Starts a new episode by setting initial state and loading environment data."""
        self.heading = heading
        self.elevation = elevation

        self.scan_ID = scan_ID
        self.viewpoint_ID = viewpoint_ID

        self.navigable = self.env_navigable[scan_ID]
        self.location = self.env_loaction[scan_ID][viewpoint_ID]
        
        self.getCandidate()  # Get initial candidate viewpoints.

    def updateGraph(self):
        """Updates the navigation graph with connections from the current viewpoint."""
        for candidate in self.candidate:  # Iterate through candidate viewpoint IDs.
            self.gmap.update_connection(self.viewpoint_ID, candidate)

    def getState(self) -> dict:
        """Returns the current state of the agent."""
        self.state = {
            'scanID': self.scan_ID,
            'viewpointID': self.viewpoint_ID,
            'heading': self.heading,
            'elevation': self.elevation,
            'viewIndex': self.get_viewIndex(),
            'candidate': self.candidate,
            'x': self.location[0],
            'y': self.location[1],
            'z': self.location[2],
        }
        return self.state
    
    def getCandidate(self):
        """Retrieves and updates candidate viewpoints for the current location."""
        self.candidate = self.navigable[self.viewpoint_ID]  # Fetch candidates.
        self.updateGraph()  # Update the exploration graph.

    def makeAction(self, next_viewpoint_ID):
        """Updates agent state by moving to the next viewpoint."""
        if next_viewpoint_ID == self.viewpoint_ID:
            return  # No movement if target is the same.
        elif next_viewpoint_ID in self.candidate:
            # Update heading and elevation based on the candidate viewpoint.
            self.heading = self.candidate[next_viewpoint_ID]['heading']
            self.elevation = self.candidate[next_viewpoint_ID]['elevation']

        self.viewpoint_ID = next_viewpoint_ID  # Move to the new viewpoint.
        self.getCandidate()  # Update available next viewpoints.

    def get_viewIndex(self):
        return int((math.degrees(self.heading) + 15) // 30 + 12 * ((math.degrees(self.elevation) + 15) // 30 + 1))


class EnvBatch(object):
    """A simple wrapper for a batch of MatterSim environments,
       using discretized viewpoints and pretrained features"""

    def __init__(self, feat_db=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feat_db: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """

        self.feat_db = feat_db

        env_data = self.load_env_data()
        
        self.sims = []
        for _ in range(batch_size):
            sim = Simulator(env_data)
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        """Creates a unique ID from scan and viewpoint IDs."""
        return f"{scanId}_{viewpointId}"

    def newEpisodes(self, scanIds, viewpointIds, headings):
        """Starts new episodes for each simulator in the batch."""
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0) # Initialize each simulator.

    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((36, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for _, sim in enumerate(self.sims):
            state = sim.getState()

            feature = self.feat_db.get_image_observation(state["scanID"], state["viewpointID"])
            feature_states.append((feature, state))
        return feature_states

    def makeActions(self, next_viewpoint_IDs):
        ''' Take an action using the full state dependent action interface (with batched input)'''
        for i, next_viewpoint_ID in enumerate(next_viewpoint_IDs):
            self.sims[i].makeAction(next_viewpoint_ID)

    def load_env_data(self):
        """
        Get the navigable and location json file from huggingface
        """
        env_data = {}

        env_data['navigable'] = load_hub_data("billzhao1030/MP3D", "navigable.json")
        env_data['location'] = load_hub_data("billzhao1030/MP3D", "location.json")

        return env_data


class R2RNavBatch(object):
    ''' Implements the navigation task with discrete viewpoints and pretrained features. '''

    def __init__(
        self, 
        view_db, 
        instr_data: str, 
        connectivity_dir: str, 
        navigable_dir: str,
        batch_size=1, 
        seed=0, 
        name=None,
    ):
        """Initializes the batch environment for navigation."""
        self.env = EnvBatch(navigable_dir, feat_db=view_db, batch_size=batch_size)
        self.data = instr_data  # List of instruction dictionaries.
        self.scans = set(x['scan'] for x in self.data)  # Unique scan IDs.
        self.connectivity_dir = connectivity_dir  # Directory containing connectivity graphs.
        self.batch_size = batch_size
        self.name = name  # Name of the dataset split.

        self.gt_trajs = self._get_gt_trajs(self.data)  # Ground truth trajectories for evaluation.

        self.seed = seed
        random.seed(self.seed)  # Seed for shuffling data.
        random.shuffle(self.data)  # Shuffle the instruction data.

        self.ix = 0  # Index for iterating through data.
        self._load_nav_graphs()  # Load navigation graphs and shortest paths.

        self.buffered_state_dict = {}  # Buffer for episode states (potentially unused in this snippet).
        print(f'{self.__class__.__name__} loaded with {len(self.data)} instructions, using splits: {self.name}')

    def _get_gt_trajs(self, data):
        """Extracts ground truth scan and path for each instruction."""
        gt_trajs = {
            x['instr_id']: (x['scan'], x['path'])
            for x in data if len(x['path']) > 1  # Only include instructions with a path.
        }
        return gt_trajs

    def size(self):
        """Returns the total number of instructions."""
        return len(self.data)

    def _load_nav_graphs(self):
        """Loads connectivity graphs, shortest paths, and distances for each scan."""
        print(f'Loading navigation graphs for {len(self.scans)} scans')
        self.graphs = load_nav_graphs(self.connectivity_dir, self.scans)  # ScanId -> Graph
        self.shortest_paths = {}  # ScanId -> (ViewId -> (ViewId -> [path]))
        for scan, G in self.graphs.items():
            self.shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))  # Compute all shortest paths.
        self.shortest_distances = {}  # ScanId -> (ViewId -> (ViewId -> distance))
        for scan, G in self.graphs.items():
            self.shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G)) # Compute all shortest distances.

    def _next_minibatch(self, batch_size=None, **kwargs):
        """Loads the next minibatch of instructions into 'self.batch'."""
        if batch_size is None:
            batch_size = self.batch_size

        batch = self.data[self.ix: self.ix + batch_size]
        if len(batch) < batch_size:
            random.shuffle(self.data)
            self.ix = batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += batch_size
        self.batch = batch  # Current minibatch of instructions.

    def reset_epoch(self, shuffle=False):
        ''' Resets the data index to the beginning of the epoch. For testing, shuffle data if requested. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _get_obs(self):
        """Gets observations for the current batch of environments."""
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]

            ob = {
                'obs' : feature["img_obs"],  # Image features.
                'obs_summary' : feature["summary"],  # Summary features.
                'map' : feature["map"],  # Map features.
                'instr_id' : item['instr_id'],  # Unique instruction ID.
                'scan' : state['scanID'],  # Current scan ID.
                'viewpoint' : state['viewpointID'],  # Current viewpoint ID.
                'heading' : state['heading'],  # Current heading.
                'elevation' : state['elevation'],  # Current elevation.
                'candidate': state['candidate'],  # Available next viewpoints.
                'instruction' : item['instruction'],  # Text instruction.
                'gt_path' : item['path'],  # Ground truth path.
                'path_id' : item['path_id']  # Unique path ID.
            }
            # Negative distance to the goal as reward. Multiple GT end viewpoints in REVERIE.
            if ob['instr_id'] in self.gt_trajs:
                ob['distance'] = self.shortest_distances[ob['scan']][ob['viewpoint']][item['path'][-1]]
            else:
                ob['distance'] = 0

            obs.append(ob)
        return obs

    def reset(self, **kwargs):
        ''' Loads a new minibatch of episodes and resets the environments. '''
        self._next_minibatch(**kwargs)

        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)  # Initialize environments.
        return self._get_obs()

    def step(self, next_viewpoint_IDs):
        ''' Takes action in the environments (same interface as makeActions). '''
        self.env.makeActions(next_viewpoint_IDs)
        return self._get_obs()

    ############### Nav Evaluation ###############
    def _get_nearest(self, shortest_distances, goal_id, path):
        """Finds the viewpoint in the predicted path closest to the goal."""
        near_id = path[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in path:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id

    def _eval_item(self, scan, pred_path, gt_path):
        """Evaluates a single predicted trajectory against the ground truth."""
        scores = {}

        shortest_distances = self.shortest_distances[scan]

        path = sum(pred_path, [])  # Flatten the predicted path.
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        nearest_position = self._get_nearest(shortest_distances, gt_path[-1], path)

        scores['nav_error'] = shortest_distances[path[-1]][gt_path[-1]]  # Distance to goal.
        scores['oracle_error'] = shortest_distances[nearest_position][gt_path[-1]]  # Oracle distance to goal.

        scores['action_steps'] = len(pred_path) - 1  # Number of actions taken.
        scores['trajectory_steps'] = len(path) - 1  # Number of viewpoints in the trajectory.
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])]) # Length of the predicted trajectory.

        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])]) # Length of the ground truth trajectory.

        scores['success'] = float(scores['nav_error'] < ERROR_MARGIN)  # Navigation success.
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01) # Success weighted by path length.
        scores['oracle_success'] = float(scores['oracle_error'] < ERROR_MARGIN) # Oracle success.

        scores.update(
            cal_dtw(shortest_distances, path, gt_path, scores['success'], ERROR_MARGIN) # DTW metrics.
        )
        scores['CLS'] = cal_cls(shortest_distances, path, gt_path, ERROR_MARGIN) # Coverage-weighted Levenshtein Similarity.

        return scores

    def eval_metrics(self, preds):
        ''' Evaluates agent trajectories based on proximity to the goal. '''
        print(f'eval {len(preds)} predictions')

        metrics = defaultdict(list)
        for item in preds:
            instr_id = item['instr_id']
            traj = item['trajectory']
            scan, gt_traj = self.gt_trajs[instr_id]
            traj_scores = self._eval_item(scan, traj, gt_traj)
            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)

        avg_metrics = {
            'action_steps': np.mean(metrics['action_steps']),
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'nav_error': np.mean(metrics['nav_error']),
            'oracle_error': np.mean(metrics['oracle_error']),
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
            'nDTW': np.mean(metrics['nDTW']) * 100,
            'SDTW': np.mean(metrics['SDTW']) * 100,
            'CLS': np.mean(metrics['CLS']) * 100,
        }
        return avg_metrics, metrics