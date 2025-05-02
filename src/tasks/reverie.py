import json
import copy
import numpy as np
from tqdm import tqdm
from logging import Logger
from omegaconf import DictConfig
from typing import List, Dict
from .mp3d_dataset import MP3DDataset
from collections import defaultdict

ERROR_MARGIN = 3.0

class REVERIEDataset(MP3DDataset):
    name = "reverie"