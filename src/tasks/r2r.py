import json
import numpy as np
from tqdm import tqdm
from .mp3d_dataset import MP3DDataset

from collections import defaultdict
ERROR_MARGIN = 3.0

class R2RDataset(MP3DDataset):
    name = "r2r"

    