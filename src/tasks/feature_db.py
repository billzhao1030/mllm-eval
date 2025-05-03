import os
import json
from PIL import Image

class ImageObservationDB(object):
    def __init__(self, img_obs_dir, img_obs_sum_dir, map_dir):
        self.image_obs_dir = img_obs_dir
        self.image_obs_sum_dir = img_obs_sum_dir
        self.map_dir = map_dir
        self._obs_store = {}

    def get_image_observation(self, scan, viewpoint):
        key = "%s_%s" % (scan, viewpoint)
        if key in self._obs_store:
            return self._obs_store[key]
        
        # Initialize the key entry once
        self._obs_store[key] = {'img_obs': [], 'summary': None, 'map': None}
        
        # Load image observation
        image_list = []
        for i in range(4):
            image_path = os.path.join(self.image_obs_dir, scan, f"{viewpoint}_{i}.png")
            image = Image.open(image_path)
            image_list.append(image)
        self._obs_store[key]['img_obs'] = image_list

        # Load image observation summary for history if available
        if self.image_obs_sum_dir:
            with open(os.path.join(self.image_obs_sum_dir, f"{scan}_summarized.json"), "r") as f:
                self._obs_store[key]['summary'] = json.load(f)[viewpoint]

        # Load map if available
        if self.map_dir:
            # TODO
            pass

        return self._obs_store[key]

def create_observation_db(config):
    img_obs_dir = config.environment.obs_dir
    img_obs_sum_dir = config.environment.obs_sum_dir
    map_dir = config.environment.map_dir

    return ImageObservationDB(img_obs_dir, img_obs_sum_dir, map_dir)

