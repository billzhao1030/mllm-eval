
import torch
import copy
from logging import Logger
from omegaconf import DictConfig

from utils.data import load_hub_data, load_graph
from .feature_db import create_observation_db
from tasks import load_dataset

def create_environment(logger):
    logger.info("Loading simulation envrionment - MP3D ...")

    navigable_data = load_hub_data("billzhao1030/MP3D", "navigable.json")
    location_data = load_hub_data("billzhao1030/MP3D", "location.json")
    scans = load_hub_data("billzhao1030/MP3D", "scans.txt", extension="txt")

    graphs, shortest_paths, shortest_distance = load_graph(location_data, navigable_data, scans)

    logger.info(f"Loaded {len(scans)} scans")

    environment = {
        "graphs": graphs,
        "shortest_path": shortest_paths,
        "shortest_distance": shortest_distance,
        "navigable": navigable_data,
        "location": location_data
    }

    return environment


def create_dataloader(
    config: DictConfig, 
    logger: Logger, 
    environment,
    device: torch.device
):

    task_cfg = copy.deepcopy(config.task)

    # Create image observation
    image_obs_db = create_observation_db(config)

    dataloaders = {}

    # Load datasets
    task_list = copy.deepcopy(task_cfg.val_source)

    for k, task in enumerate(task_list):
        splits = task_cfg.eval_splits[task]
        if isinstance(splits, str):
                splits = [splits]

        for split in splits:
            dataset = load_dataset(
                name=task.lower(), 
                config=config, 
                split=split, 
                environment=environment, 
                logger=logger, 
                source=task
            )

            task_feat_db = {}



    


    

