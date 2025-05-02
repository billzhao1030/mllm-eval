import copy
from logging import Logger
from omegaconf import DictConfig
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, SequentialSampler
from typing import List, Dict, Tuple, Union, Iterator

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
            # Get annotation data
            dataset = load_dataset(
                name=task.lower(), 
                config=config, 
                split=split, 
                environment=environment, 
                logger=logger, 
                task=task
            )

            # Set observation database
            dataset.init_obs_db(image_obs_db)

            task_loader, pre_epoch = build_dataloader(
                 dataset=dataset,
                 distributed=config.distributed.distributed,
                 batch_size=config.experiment.batch_size,
                 num_workers=config.experiment.workers
            )

            dataloader_name = f"{task}.{split}"
            dataloaders[dataloader_name] = PrefetchLoader(task_loader, device=device)

    return dataloaders

def build_dataloader(dataset, distributed, batch_size, num_workers):
    if distributed:
        size = dist.get_world_size()
        sampler = DistributedSampler(
            dataset, num_replicas=size, rank=dist.get_rank(), shuffle=False
        )
        pre_epoch = sampler.set_epoch
    else:
        sampler = SequentialSampler(dataset)

        size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        pre_epoch = lambda e: None

    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset.collate_batch,
    )
    loader.num_batches = len(loader)

    return loader, pre_epoch

def move_to_cuda(batch: Union[List, Tuple, Dict, torch.Tensor], device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, list):
        return [move_to_cuda(t, device) for t in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_cuda(t, device) for t in batch)
    elif isinstance(batch, dict):
        return {n: move_to_cuda(t, device) for n, t in batch.items()}
    return batch


class PrefetchLoader(object):
    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.device = device
        self.num_batches = self.loader.num_batches

    def get_dataset(self):
        return self.loader.dataset

    def __iter__(self):
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return
        self.batch = move_to_cuda(self.batch, self.device)

    def next(self, it):
        batch = self.batch
        self.preload(it)
        return batch

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method 
          

    


    

