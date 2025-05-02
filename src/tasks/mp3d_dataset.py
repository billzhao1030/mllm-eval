import os
from logging import Logger
from omegaconf import DictConfig
from typing import List, Dict
from .base_dataset import BaseDataset
from huggingface_hub import hf_hub_download

class MP3DDataset(BaseDataset):
    TASK_ID = {
        'R2R': 0,
        'REVERIE': 1,
        'SOON': 2,
        'RXR-EN': 3,
        'CVDN': 4,
    }

    def __init__(
        self,
        config: DictConfig,
        split: str,
        environment: Dict,
        logger: Logger,
        task: str
    ):
        super().__init__()

        self.config = config
        self.logger = logger
        self.debug = config.experiment.debug
        self.task = task
        self.split = split
        
        self.batch_size = config.experiment.batch_size
        self.seed = config.experiment.seed

        self.feat_db = None

        # Load MP3D dataset
        self._load_data(config.experiment.data_dir)

    def _load_data(self, data_dir):
        # Set file extension
        ext = ".jsonl" if self.task == "RXR_EN" else ".json"
        file_keys = [
            "test_enc", "train_enc", "val_seen_enc",
            "val_train_seen_enc", "val_unseen_enc", "val_unseen_subset_enc"
        ]

        # Create local target folder
        target_folder = os.path.join(data_dir, self.task)
        os.makedirs(target_folder, exist_ok=True)

        anno_file = f"{self.split}_enc{ext}"
        local_path = os.path.join(target_folder, anno_file)

        if os.path.exists(local_path):
            self.logger.info(f"Found {anno_file}, skipping download.")
        else:
            self._download_data(anno_file, local_path)

        self.logger.info(f"Loading R2R data from {anno_file}")
            

    def _download_data(self, filename, local_path):
        hub_filename = f"{self.task}/{filename}"
        downloaded_path = hf_hub_download(
            repo_id="billzhao1030/VLN_annotations",
            filename=hub_filename,
            repo_type="dataset"
        )

        with open(downloaded_path, "rb") as src, open(local_path, "wb") as dst:
            dst.write(src.read())
        self.logger.info(f"Downloaded {hub_filename} -> {local_path}")
            