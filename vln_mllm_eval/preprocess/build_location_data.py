import os
import json
from huggingface_hub import HfApi

from huggingface_util import upload_file


def build_viewpoint_locations_json(connectivity_dir, output_path):
    """
    Reads connectivity JSON files, extracts viewpoint locations, and saves
    this information to a JSON file.

    Args:
        connectivity_dir (str): Path to the directory containing the
                                 '_connectivity.json' files.
        output_path (str): Path to the JSON file where the viewpoint locations
                             will be saved.
    """
    all_locations = {}
    scan_files = [f for f in os.listdir(connectivity_dir) if f.endswith('_connectivity.json')]
    scan_files.sort()

    for scan_file in scan_files:
        scan_id = scan_file.replace('_connectivity.json', '')
        connectivity_path = os.path.join(connectivity_dir, scan_file)
        room_locations = {}

        try:
            with open(connectivity_path, 'r') as f:
                connectivity_data = json.load(f)

            for connection in connectivity_data:
                image_id = connection['image_id']
                pose = connection['pose']
                location = [pose[3], pose[7], pose[11]]
                room_locations[image_id] = location

            all_locations[scan_id] = room_locations
            print(f"Processed scan: {scan_id} - Found {len(room_locations)} viewpoints.")

        except FileNotFoundError:
            print(f"Error: Connectivity file not found at {connectivity_path}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {connectivity_path}")
        except KeyError as e:
            print(f"Error: Key '{e}' not found in {connectivity_path}. Check the connectivity file structure.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {connectivity_path}: {e}")

    try:
        with open(output_path, 'w') as outfile:
            json.dump(all_locations, outfile, indent=4)
        print(f"\nViewpoint locations saved to: {output_path}")
    except Exception as e:
        print(f"Error writing to output file {output_path}: {e}")


if __name__ == "__main__":
    username = "billzhao1030"
    connectivity_directory = '../datasets/R2R/connectivity'
    output_file = '../datasets/R2R/location.json'
    repo_id = f"MP3D"

    # Build the navigable JSON files
    build_viewpoint_locations_json(connectivity_directory, output_file)

    # Upload the entire navigable file to Hugging Face
    upload_file(output_file, username, repo_id)

    print("Script finished: Viewpoint Location data built and uploaded to Hugging Face Hub.")