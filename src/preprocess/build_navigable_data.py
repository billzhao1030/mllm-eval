import os
import json
import math
import MatterSim
import numpy as np

from collections import defaultdict
from huggingface_util import upload_file

def new_simulator(connectivity_dir, scan_data_dir=None):
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
    sim.initialize()

    return sim

def build_navigable_json(connectivity_dir, output_path):
    """
    Builds a single navigable JSON file with scan ID as the key, and the
    navigable data for that scan as the value.

    Args:
        connectivity_dir (str): Path to the directory containing the
                                 '_connectivity.json' files.
        output_path (str): Path to the single JSON file where all
                             navigable data will be saved.
    """
    sim = new_simulator(connectivity_dir)
    all_scans_navigable_data = {}

    scan_files = [f for f in os.listdir(connectivity_dir) if f.endswith('_connectivity.json')]
    scan_files.sort()

    for scan_file in scan_files:
        scan_id = scan_file.replace('_connectivity.json', '')
        connectivity_path = os.path.join(connectivity_dir, scan_file)
        scan_navigable_data = defaultdict(dict)

        try:
            with open(connectivity_path, 'r') as f:
                connectivity_list = json.load(f)

            for connection in connectivity_list:
                if not connection['included']:
                    continue
                viewpoint_id = connection['image_id']

                for view_id in range(36):
                    if view_id == 0:
                        sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
                    elif view_id % 12 == 0:
                        sim.makeAction([0], [1.0], [1.0])
                    else:
                        sim.makeAction([0], [1.0], [0])

                    state = sim.getState()[0]

                    for nav_loc in state.navigableLocations[1:]:
                        loc_heading = state.heading + nav_loc.rel_heading
                        loc_elevation = state.elevation + nav_loc.rel_elevation
                        distance = np.sqrt(nav_loc.rel_heading ** 2 + nav_loc.rel_elevation ** 2)

                        if nav_loc.viewpointId not in scan_navigable_data[viewpoint_id] or distance < scan_navigable_data[viewpoint_id][nav_loc.viewpointId]['ang_dis']:
                            scan_navigable_data[viewpoint_id][nav_loc.viewpointId] = {
                                "heading": loc_heading,
                                "elevation": loc_elevation,
                                "ang_dis": distance,
                                "distance": nav_loc.rel_distance
                            }

            all_scans_navigable_data[scan_id] = dict(scan_navigable_data)
            print(f"Processed scan: {scan_id}")

        except FileNotFoundError:
            print(f"Error: Connectivity file not found for scan {scan_id}")
        except Exception as e:
            print(f"An error occurred while processing scan {scan_id}: {e}")

    try:
        with open(output_path, 'w') as outfile:
            json.dump(all_scans_navigable_data, outfile, indent=4)
        print(f"All scans processed and saved to: {output_path}")
    except Exception as e:
        print(f"Error writing to output file: {e}")

if __name__ == "__main__":
    username = "billzhao1030"
    connectivity_directory = '../datasets/R2R/connectivity'
    output_file = '../datasets/R2R/navigable.json'
    repo_id = f"MP3D"

    # Build the navigable JSON files
    build_navigable_json(connectivity_directory, output_file)

    # Upload the entire navigable file to Hugging Face
    upload_file(output_file, username, repo_id)

    print("Script finished: Navigable data built and uploaded to Hugging Face Hub.")