import os
import json
import math

import numpy as np
import networkx as nx

from PIL import Image
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, TypeVar, NamedTuple, Union

# class ImageFeaturesDB(object):
#     def __init__(self, img_ft_file, image_feat_size):
#         self.image_feat_size = image_feat_size
#         self.img_ft_file = img_ft_file
#         self._feature_store = {}

#     def get_image_feature(self, scan, viewpoint):
#         key = '%s_%s' % (scan, viewpoint)
#         if key in self._feature_store:
#             ft = self._feature_store[key]
#         else:
#             with h5py.File(self.img_ft_file, 'r') as f:
#                 ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
#                 self._feature_store[key] = ft
#         return ft

class ImageObservationsDB(object):
    def __init__(self, img_obs_dir, img_obs_sum_dir, map_dir):
        self.img_obs_dir = img_obs_dir
        self.img_obs_sum_dir = img_obs_sum_dir
        self.map_dir = map_dir
        self._obs_store = {}

    def get_image_observation(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if key in self._obs_store:
            obs = self._obs_store[key]
        else:
            # Load image observation
            self._obs_store[key] = {}
            images_list = []
            for i in range(4):
                # image = cv2.imread(os.path.join(self.img_obs_dir, scan, f"{viewpoint}_{i}.png"))
                # # TODO: check cv2 color conversion
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_path = os.path.join(self.img_obs_dir, scan, f"{viewpoint}_{i}.png")
                image = Image.open(image_path)
                images_list.append(image)
            self._obs_store[key]['img_obs'] = images_list
            # Load image observation summary for history
            with open(os.path.join(self.img_obs_sum_dir, f'{scan}_summarized.json'), 'r') as f:
                obs_sum = json.load(f)[viewpoint]
                self._obs_store[key]['summary'] = obs_sum
            if self.map_dir is not None:
                pass
            else:
                self._obs_store[key]['map'] = None
            obs = self._obs_store[key]
        return obs

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


def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

def new_simulator(connectivity_dir, scan_data_dir=None):
    import MatterSim

    # Simulator image parameters
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60

    sim = MatterSim.Simulator()
    if scan_data_dir:
        sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()

    return sim

def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading), math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)

def get_point_angle_feature(sim, angle_feat_size, baseViewId=0):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    base_elevation = (baseViewId // 12 - 1) * math.radians(30)

    for ix in range(36):
        if ix == 0:
            sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading
        elevation = state.elevation - base_elevation

        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    return feature

def get_all_point_angle_feature(sim, angle_feat_size):
    return [get_point_angle_feature(sim, angle_feat_size, baseViewId) for baseViewId in range(36)]

def load_instr_datasets(anno_dir, dataset, splits):
    data = []
    for split in splits:
        filepath = os.path.join(anno_dir, f'{split}.json')
        with open(filepath) as f:
            new_data = json.load(f)

        data += new_data

    return data

def construct_instrs(anno_dir, dataset, splits):
    data = []
    if "instr" in splits[0]:
        return load_instr_datasets(anno_dir, dataset, splits)

    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits)):
        # Split multiple instructions into separate entries 
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)
            new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
            new_item['instruction'] = instr
            del new_item['instructions']
            del new_item['instr_encodings']
            data.append(new_item)
    return data
