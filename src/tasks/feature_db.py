import os
import io
import json
from PIL import Image
import matplotlib.pyplot as plt

from utils.data import download_mp3d_observations

class ImageObservationDB(object):
    def __init__(
        self, 
        img_obs_dir,
        img_marker_cap_dir, 
        img_obs_sum_dir, 
        map_dir
    ):
        self.image_obs_dir = img_obs_dir
        self.marker_caption_dir = img_marker_cap_dir
        self.image_obs_sum_dir = img_obs_sum_dir
        self.map_dir = map_dir
        self._obs_store = {}

        self.directions = ["left", "front", "right", "back"]

    def get_image_observation(self, scan, viewpoint):
        key = "%s_%s" % (scan, viewpoint)
        if key in self._obs_store:
            return self._obs_store[key]
        
        # Initialize the key entry once
        self._obs_store[key] = {
            'img_obs': {}, 
            'caption': {}, 
            'summary': None, 
            'map': None,
            'id_viewpoint': self.map_caption_indices_to_viewpoints(scan, viewpoint)
        }
        
        # Load image observation
        for direction in self.directions:
            image_filename = f"{scan}_{viewpoint}_{direction}.png"
            image_path = os.path.join(self.image_obs_dir, image_filename)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Missing image: {image_path}")
            image = Image.open(image_path).convert('RGB')
            self._obs_store[key]['img_obs'][direction] = image

        # Load the caption of the markers
        with open(self.marker_caption_dir, 'r') as f:
            self._obs_store[key]['caption'] = json.load(f)[key]

        # Load image observation summary for history if available
        if self.image_obs_sum_dir:
            with open(os.path.join(self.image_obs_sum_dir, f"{scan}_summarized.json"), "r") as f:
                self._obs_store[key]['summary'] = json.load(f)[viewpoint]

        # Load map if available
        if self.map_dir:
            # TODO
            pass

        return self._obs_store[key]

    def map_caption_indices_to_viewpoints(self, scan_id, viewpoint_id):
        with open("../data/MP3D/navigable.json", 'r') as f:
            navigable = json.load(f)

        # Get the dictionary of navigable viewpoints for the given scan and viewpoint
        navigable_points = navigable.get(scan_id, {}).get(viewpoint_id, {})

        # Map caption indices to viewpoint IDs based on order
        caption_id_to_viewpoint = {
            str(i + 1): vp_id
            for i, vp_id in enumerate(navigable_points.keys())
        }

        return caption_id_to_viewpoint
    
    def display_four_images_with_heading(img_obs, heading_deg=0):
        # Normalize heading to be within 0 to 359 degrees
        normalized_heading = heading_deg % 360
        if normalized_heading < 0:
            normalized_heading += 360

        if (normalized_heading >= 315 and normalized_heading < 360) or \
        (normalized_heading >= 0 and normalized_heading < 45):
            current_front_key = 'front'
        elif normalized_heading >= 45 and normalized_heading < 135:
            current_front_key = 'right'
        elif normalized_heading >= 135 and normalized_heading < 225:
            current_front_key = 'back'
        elif normalized_heading >= 225 and normalized_heading < 315:
            current_front_key = 'left'

        global_order_keys = ['left', 'front', 'right', 'back']
        idx_current_front = global_order_keys.index(current_front_key)

        display_keys = [
            global_order_keys[(idx_current_front - 1 + 4) % 4], # Agent's Left
            global_order_keys[idx_current_front],               # Agent's Front
            global_order_keys[(idx_current_front + 1) % 4],     # Agent's Right
            global_order_keys[(idx_current_front + 2) % 4]      # Agent's Back
        ]

        TITLE_FONT_SIZE = 18
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

        for i, key_to_display in enumerate(display_keys):
            img_to_show = img_obs.get(key_to_display)

            if img_to_show is not None and isinstance(img_to_show, Image.Image):
                axes[i].imshow(img_to_show)
                axes[i].set_title(global_order_keys[i], fontsize=TITLE_FONT_SIZE)
                axes[i].axis('off')
            else:
                print(f"Warning: Image for global '{key_to_display}' is missing or not a PIL Image object.")
                axes[i].set_title(f'No {global_order_keys[i]} Image', fontsize=TITLE_FONT_SIZE)
                axes[i].axis('off')

        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)  # Close the figure to free memory
        buf.seek(0)

        # Convert buffer to PIL Image
        img_result = Image.open(buf)
        return img_result

def create_observation_db(config, logger):
    img_obs_dir = config.environment.obs_dir
    caption_dir = config.environment.caption_dir
    img_obs_sum_dir = config.environment.obs_sum_dir
    map_dir = config.environment.map_dir

    # download_mp3d_observations(logger)

    return ImageObservationDB(
        img_obs_dir, 
        caption_dir,
        img_obs_sum_dir, 
        map_dir
    )

