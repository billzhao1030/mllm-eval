import os
from tqdm import tqdm
from huggingface_util import load_hub_data
from datasets import load_dataset

# Setup
scan_ids = load_hub_data("billzhao1030/MP3D", "scans.txt", extension="txt")
dataset_repo = "billzhao1030/MP3D_feature"
save_dir = "../data/observations"

# Download scan-by-scan
for scan_id in tqdm(scan_ids, desc="Processing scans"):

    # Create the scan folder locally
    scan_folder = os.path.join(save_dir, scan_id)
    os.makedirs(scan_folder, exist_ok=True)

    # If already exists and non-empty, skip
    if os.listdir(scan_folder):
        print(f"Scan {scan_id} already downloaded, skipping.")
        continue

    print(f"\nDownloading split (scan): {scan_id}")
    scan_dataset = load_dataset(dataset_repo, split=scan_id, streaming=True)

    for item in tqdm(scan_dataset, desc=f"Downloading {scan_id}", leave=False):

        viewpoint_id = item["viewpoint_id"]

        # Save four images
        for i in range(4):
            img = item[f"image_{i}"]  # <--- directly get the PIL image!
            img_save_path = os.path.join(scan_folder, f"{viewpoint_id}_{i}.png")
            img.save(img_save_path)

    print(f"✅ Finished scan {scan_id}.\n")

print("All scans finished")