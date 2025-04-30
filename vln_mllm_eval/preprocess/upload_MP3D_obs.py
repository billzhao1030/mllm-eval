import os
import pandas as pd
from datasets import Dataset, Features, Value, Image
from huggingface_hub import HfApi

def load_single_scan(scan_id, observation_dir):
    scan_folder = os.path.join(observation_dir, scan_id)
    rows = []
    
    if not os.path.isdir(scan_folder):
        return None  # Skip non-folders

    viewpoint_groups = {}
    for file in os.listdir(scan_folder):
        if file.endswith(".png"):
            try:
                viewpoint_id, direction = file[:-4].split('_')
                direction = int(direction)
            except ValueError:
                print(f"Skipping malformed file name: {file}")
                continue

            key = (scan_id, viewpoint_id)
            if key not in viewpoint_groups:
                viewpoint_groups[key] = [None] * 4

            img_path = os.path.join(scan_folder, file)
            with open(img_path, "rb") as f:
                viewpoint_groups[key][direction] = f.read()

    for (scan_id, viewpoint_id), images in viewpoint_groups.items():
        if all(images):
            rows.append({
                "scan_id": scan_id,
                "viewpoint_id": viewpoint_id,
                "image_0": images[0],
                "image_1": images[1],
                "image_2": images[2],
                "image_3": images[3],
            })

    if rows:
        return pd.DataFrame(rows)
    else:
        return None
    
if __name__ == "__main__":
    api = HfApi()
    api.create_repo(repo_id="MP3D_feature", repo_type="dataset", exist_ok=True)

    # Step 1: setup
    observation_dir = "../datasets/observations"

    features = Features({
        "scan_id": Value("string"),
        "viewpoint_id": Value("string"),
        "image_0": Image(),
        "image_1": Image(),
        "image_2": Image(),
        "image_3": Image()
    })

    # Step 2: build dataset 

    # Choose how many scans you want to upload at once (small batches recommended)
    scan_ids = os.listdir(observation_dir)
    scan_ids.sort()

    for i, scan_id in enumerate(scan_ids):
        print(f"Processing scan {i+1}: {scan_id}")

        df_scan = load_single_scan(scan_id, observation_dir)

        dataset = Dataset.from_pandas(df_scan, features=features)

        # Push this dataset as a new split
        dataset.push_to_hub("billzhao1030/MP3D_feature", split=scan_id)
        print(f"Uploaded scan '{scan_id}' as a split.")