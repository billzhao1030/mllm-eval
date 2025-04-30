from logging import Logger
from omegaconf import DictConfig
from typing import List, Dict
from .base_dataset import BaseDataset

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
        source: str
    ):
        super.__init__()

        self.config = config
        self.logger = logger
        self.debug = config.experiment.debug
        self.source = source
        self.split = split
        
        self.batch_size = config.experiment.batch_size
        self.seed = config.experiment.seed

        self.feat_db = None

        # Load MP3D dataset
        