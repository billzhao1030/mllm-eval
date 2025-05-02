import json
import numpy as np
from tqdm import tqdm
from .mp3d_dataset import MP3DDataset

from collections import defaultdict
ERROR_MARGIN = 3.0

class R2RDataset(MP3DDataset):
    name = "r2r"

    def load_data(self, anno_file, max_instr_len=200):
        with open(anno_file, "r") as f:
            data = json.load(f)

        new_data = []

        data = tqdm(data, desc="Loading data")
        for i, item in enumerate(data):
            sample_index = 0
            for j, instr in enumerate(item["instructions"]):
                new_item = dict(item)
                new_item['raw_idx'] = i
                new_item['sample_idx'] = sample_index
                new_item['instr_id'] = f"{self.name}_{item['path_id']}_{j}"

                new_item['instruction'] = instr
                del new_item['instructions']

                new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
                del new_item['instr_encodings']

                new_item['data_type'] = 'r2r'

                new_data.append(new_item)
                sample_index += 1

        gt_trajs = {
            x['instr_id']: (x['scan'], x['path']) \
            for x in new_data if len(x['path']) > 1
        }    

        return new_data, gt_trajs