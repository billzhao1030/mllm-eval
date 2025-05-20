import os
import pandas as pd
from datasets import Dataset, Features, Value, Image
from huggingface_hub import HfApi

# Mapping from direction string to column name
DIRECTION_MAP = {
    "left": "left",
    "front": "front",
    "right": "right",
    "back": "back"
}

def load_single_scan(scan_id, observation_dir):
    scan_folder = os.path.join(observation_dir)
    rows = {}

    for file in os.listdir(scan_folder):
        if not file.endswith(".png"):
            continue
        try:
            name = file[:-4]  # remove .png
            parts = name.split("_")
            if len(parts) < 3:
                print(f"Skipping malformed file name: {file}")
                continue
            scan_id_file = parts[0]
            viewpoint_id = parts[1]
            direction = "_".join(parts[2:])
        except Exception as e:
            print(f"Error parsing filename '{file}': {e}")
            continue

        if scan_id_file != scan_id or direction not in DIRECTION_MAP:
            continue

        key = (scan_id, viewpoint_id)
        if key not in rows:
            rows[key] = {
                "scan_id": scan_id,
                "viewpoint_id": viewpoint_id,
                "left": None,
                "front": None,
                "right": None,
                "back": None
            }

        rows[key][DIRECTION_MAP[direction]] = os.path.join(scan_folder, file)

    valid_rows = []
    for row in rows.values():
        if all(row[d] is not None for d in ["left", "front", "right", "back"]):
            valid_rows.append(row)
        else:
            print(f"Skipping incomplete viewpoint {row['viewpoint_id']} in scan {scan_id}")

    return pd.DataFrame(valid_rows) if valid_rows else None


if __name__ == "__main__":
    repo_id = "billzhao1030/MP3D_marked_obs"
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    observation_dir = "../data/marked_obs"

    features = Features({
        "scan_id": Value("string"),
        "viewpoint_id": Value("string"),
        "left": Image(),
        "front": Image(),
        "right": Image(),
        "back": Image()
    })

    scan_ids = list(set(f.split("_")[0] for f in os.listdir(observation_dir) if f.endswith(".png")))
    scan_ids.sort()

    for i, scan_id in enumerate(scan_ids):
        print(f"\n[{i+1}/{len(scan_ids)}] Processing scan: {scan_id}")
        df_scan = load_single_scan(scan_id, observation_dir)
        if df_scan is None:
            print(f"Skipping scan {scan_id} due to no valid data.")
            continue

        dataset = Dataset.from_pandas(df_scan, features=features)

        print(f"Pushing scan '{scan_id}' to Hugging Face Hub...")
        dataset.push_to_hub(repo_id, split=scan_id)
        print(f"✅ Uploaded split: {scan_id}")

    print("\n🎉 Finished uploading all scans!")
