import math
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
            'scan': self.scan_ID,
            'viewpoint': self.viewpoint_ID,
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

    def __init__(self, env_data, obs_db=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feat_db: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """

        self.obs_db = obs_db
        self.env_data = env_data
        
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

            feature = self.obs_db.get_image_observation(state["scanID"], state["viewpointID"])
            feature_states.append((feature, state))
        return feature_states

    def makeActions(self, next_viewpoint_IDs):
        ''' Take an action using the full state dependent action interface (with batched input)'''
        for i, next_viewpoint_ID in enumerate(next_viewpoint_IDs):
            self.sims[i].makeAction(next_viewpoint_ID)
