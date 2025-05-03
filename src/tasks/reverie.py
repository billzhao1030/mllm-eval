import json
import numpy as np
from tqdm import tqdm
from .mp3d_dataset import MP3DDataset
from utils.eval import *

from collections import defaultdict

ERROR_MARGIN = 3.0

class REVERIEDataset(MP3DDataset):
    name = "reverie" 

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

                new_item['data_type'] = 'reverie'

                new_data.append(new_item)
                sample_index += 1

        gt_trajs = {
            x['instr_id']: (x['scan'], x['path']) \
            for x in new_data if len(x['path']) > 1
        }    

        return new_data, gt_trajs
    
     ############### Nav Evaluation ###############

    def _get_nearest(self, shortest_distances, goal_id, path):
        near_id = path[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in path:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id
    
    def _eval_item(self, scan, pred_path, gt_path):
        """Evaluates a single predicted trajectory against the ground truth."""
        scores = {}

        shortest_distances = self.environment['shortest_distance'][scan]

        path = sum(pred_path, [])
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        nearest_position = self._get_nearest(shortest_distances, gt_path[-1], path)

        scores['nav_error'] = shortest_distances[path[-1]][gt_path[-1]]
        scores['oracle_error'] = shortest_distances[nearest_position][gt_path[-1]]

        scores['action_steps'] = len(pred_path) - 1
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])

        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])
        
        scores['success'] = float(scores['nav_error'] < ERROR_MARGIN)
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        scores['oracle_success'] = float(scores['oracle_error'] < ERROR_MARGIN)

        scores.update(
            cal_dtw(shortest_distances, path, gt_path, scores['success'], ERROR_MARGIN)
        )
        scores['CLS'] = cal_cls(shortest_distances, path, gt_path, ERROR_MARGIN)

        return scores
    
    def eval_metric(self, preds, logger):
        """Evaluates agent trajectories based on proximity to the goal."""
        logger.info('Evaluated %d predictions' % (len(preds)))
        metrics = defaultdict(list)

        for item in preds:
            instr_id = item['instr_id']
            traj = item['trajectory']
            scan, gt_traj = self.gt_trajs[instr_id]
            
            traj_scores = self._eval_item(scan, traj, gt_traj)
            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)
        
        avg_metrics = {
            'action_steps': np.mean(metrics['action_steps']),
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'nav_error': np.mean(metrics['nav_error']),
            'oracle_error': np.mean(metrics['oracle_error']),
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
            'nDTW': np.mean(metrics['nDTW']) * 100,
            'SDTW': np.mean(metrics['SDTW']) * 100,
            'CLS': np.mean(metrics['CLS']) * 100,
        }
        return avg_metrics, metrics

