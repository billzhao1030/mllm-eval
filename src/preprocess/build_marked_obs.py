import os
import math
import json
import py360convert
import numpy as np
import matplotlib.pyplot as plt
import cv2

import MatterSim

# Configuration for the simulator
VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint

WIDTH = 512
HEIGHT = 512
VFOV = 90

# Matterport3D dataset directories
connectivity_dir = 'datasets/R2R/connectivity'
scan_dir = '/home/gengze/Datasets/MP3D/v1/scans'
# Candidate viewpoints list
candidate_list_dir = '/home/gengze/Desktop/projects/versatile/de/data/simulator/mp3d_scanvp_candidates.json'

# Utility functions
def build_simulator(connectivity_dir, scan_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(False)
    sim.setElevationLimits(-math.radians(VFOV), math.radians(VFOV))
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim


def make_pano(scan_id, viewpoint_id, heading = 0, elevation = 0):
    sim = build_simulator(connectivity_dir, scan_dir)
    cube_images = []
    # Loop all discretized views from this location
    for ix in range(6):
        if ix == 0:
            # Generate Simulator with current heading
            sim.newEpisode([scan_id], [viewpoint_id], [heading], [elevation])
        elif ix == 4:
            sim.makeAction([0], [math.radians(90)], [math.radians(90)])
        elif ix == 5:
            sim.makeAction([0], [0], [math.radians(-180)])
        else:
            sim.makeAction([0], [math.radians(90)], [0])
        state = sim.getState()[0]
        image = np.array(state.rgb, copy=True)[..., ::-1]
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if ix == 1 or ix == 2:
            image = np.fliplr(image)
        elif ix == 4:
            image = np.flipud(image)
        cube_images.append(image)

    # # Convert to equirectangular
    # equirectangular = py360convert.c2e(cube_images, h=HEIGHT, w=2*WIDTH, cube_format='list')
    # # For visualization, convert to uint8
    # equirectangular = np.array(equirectangular, dtype=np.uint8)

    return state, cube_images

def show_views(scan_id, viewpoint_id, heading=0, elevation=0):
    state, cube_images = make_pano(scan_id, viewpoint_id, heading, elevation)

    front_img, right_img, back_img, left_img, up_img, down_img = cube_images
    right_img = np.fliplr(right_img)
    back_img = np.fliplr(back_img)

    # plot cube and equirectangular in one figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 10))
    axes[0].imshow(left_img)
    axes[0].set_title('Left')
    axes[1].imshow(front_img)
    axes[1].set_title('Front')
    axes[2].imshow(right_img)
    axes[2].set_title('Right')
    axes[3].imshow(back_img)
    axes[3].set_title('Back')
    plt.show()

# Helper function: Convert spherical coordinates to pixel coordinates
def spherical_to_pixel(theta, phi, d, image_width, image_height, fov_deg):
    fov_rad = np.radians(fov_deg)
    focal_length = (image_width / 2) / np.tan(fov_rad / 2)
    x = d * np.cos(phi) * np.sin(theta)
    y = d * np.sin(phi)
    z = d * np.cos(phi) * np.cos(theta)
    u = (x * focal_length) / z
    v = (y * focal_length) / z
    pixel_x = int(np.round(u + (image_width / 2)))
    pixel_y = int(np.round((image_height / 2) - v))
    return pixel_x, pixel_y

# Assign viewpoints to corresponding views
def assign_view(theta):
    angle_deg = np.degrees(theta) % 360
    if -45 <= angle_deg <= 45 or angle_deg >= 315:
        return 'front', theta
    elif 45 < angle_deg <= 135:
        return 'right', theta - np.radians(90)
    elif 135 < angle_deg <= 225:
        return 'back', theta - np.radians(180)
    else:
        return 'left', theta - np.radians(270)
    
def load_candidate_viewpoints(candidate_list_dir, scan, viewpoint):
    with open(candidate_list_dir, 'r') as f:
        candidate_dict_list = json.load(f)
    candidate_dict = candidate_dict_list[f'{scan}_{viewpoint}']

    for key, value in candidate_dict.items():
        candidate_dict[key] = {
            "normalized_heading": value[2],
            "normalized_elevation": value[3],
            "distance": value[1],
        }
    return candidate_dict

# Main function to mark viewpoints
def mark_viewpoints(viewpoints, images, fov_deg=90):
    for idx, (theta, phi, d) in enumerate(viewpoints):
        view, adj_theta = assign_view(theta)
        image = images[view]
        h, w, _ = image.shape
        px, py = spherical_to_pixel(adj_theta, phi, d, w, h, fov_deg)

        # Mark viewpoint with circle and index label
        cv2.circle(image, (px, py), 20, (0, 255, 0), -1)
        cv2.putText(image, f"{idx+1}", (px - 7, py + 7), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 255), 2)
    return images

cur_scan = "1LXtFkjw3qL"
cur_viewpoint = "0b22fa63d0f54a529c525afbf2e8bb25"
cur_heading = math.radians(0)
cur_elevation = 0
cur_candidates = []

state, cube_images= make_pano(cur_scan, cur_viewpoint, heading=cur_heading, elevation=cur_elevation)

front_img, right_img, back_img, left_img, up_img, down_img = cube_images
right_img = np.fliplr(right_img)
back_img = np.fliplr(back_img)

view_images = {
    'front': front_img.astype(np.uint8),
    'right': right_img.astype(np.uint8),
    'back': back_img.astype(np.uint8),
    'left': left_img.astype(np.uint8),
}

# Mark candidate viewpoints
mark_viewpoints(cur_candidates, view_images)

# plot cube and equirectangular in one figure
fig, axes = plt.subplots(1, 4, figsize=(20, 10))
axes[0].imshow(view_images['left'])
axes[0].set_title('Left')
axes[1].imshow(view_images['front'])
axes[1].set_title('Front')
axes[2].imshow(view_images['right'])
axes[2].set_title('Right')
axes[3].imshow(view_images['back'])
axes[3].set_title('Back')
plt.show()