import os
import torch
from logging import Logger
from omegaconf import DictConfig
from typing import List, Dict
from collections import defaultdict
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
        msg = self._load_data(config.experiment.data_dir)

        self.buffered_state_dict = {}

        all_graphs = environment['graphs']
        all_shortest_paths = environment['shortest_path']
        all_shortest_distances = environment['shortest_distance']

        self.environment = {
            "graphs": {scan: all_graphs[scan] for scan in self.scans},
            "shortest_path": {scan: all_shortest_paths[scan] for scan in self.scans},
            "shortest_distance": {scan: all_shortest_distances[scan] for scan in self.scans},
            "navigable": environment['navigable'],
            "location": environment['location']
        }

        logger.info(f"{task}: {self.__class__.__name__} {split} split loaded")
        logger.info(msg)


    def _load_data(self, data_dir):
        self.data = dict()
        msg = ""

        # Set file extension
        ext = ".jsonl" if self.task == "RXR_EN" else ".json"

        # Create local target folder
        target_folder = os.path.join(data_dir, self.task)
        os.makedirs(target_folder, exist_ok=True)

        anno_file = f"{self.split}_enc{ext}"
        local_path = os.path.join(target_folder, anno_file)

        # Download data if not exist in local
        if os.path.exists(local_path):
            self.logger.info(f"Found {anno_file}, skipping download.")
        else:
            self._download_data(anno_file, local_path)

        self.logger.info(f"Loading {self.task} data from {anno_file}")

        self.data, self.gt_trajs = self.load_data(local_path)

        # Set up scans
        self.scans = set([x['scan'] for x in self.data])
        msg += f"\n- Dataset: load {self.split} split: {len(self.data)} samples in total"
        msg += f"\n- Dataset: load {self.split} split: {len(self.scans)} scans in total"
            
        return msg

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

    def __len__(self):
        return len(self.data)

    def init_obs_db(self, obs_db):
        self.obs_db = obs_db

    @staticmethod
    def collate_batch(
        batch_list: List[Dict],
        _unused: bool = False,
    ) -> Dict:
        # batch list is a list of dictionaries from __getitem__
        data_dict = defaultdict(list)

        # collate the data dictionaries from the batch list into a single dictionary
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        for key, val in data_dict.items():
            try:
                if key in ['NotImplemented']:
                    ret[key] = torch.stack(val, 0)
                else:
                    ret[key] = val
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret
            