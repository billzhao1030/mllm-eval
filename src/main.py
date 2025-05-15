import os
import numpy as np
import torch
import random
import argparse

# from argument import parse_args
from omegaconf import OmegaConf
from utils.distributed import world_info_from_env, init_distributed_device
from tasks.loader import create_environment, create_val_env
from utils.common_utils import setup_logger, log_config_to_file


def setup_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment runner with OmegaConf")

    parser.add_argument(
        "--config_dir", 
        type=str, 
        default="configs/experiment.yaml",
        help="Path to the experiment config file"
    )

    # Option to override specific configurations using key-value pairs
    parser.add_argument(
        "--options",
        nargs="+",
        help="Override some settings in the config. Use the key-value pair in xxx=yyy format. Example: --options training.learning_rate=0.01"
    )

    args = parser.parse_args()

    # Load the default configuration
    default_config_path = 'configs/default.yaml'
    if not os.path.exists(default_config_path):
        raise FileNotFoundError(f"Default config file not found: {default_config_path}")
    default_config = OmegaConf.load(default_config_path)

    # Load the experiment-specific configuration
    if not os.path.exists(args.config_dir):
        raise FileNotFoundError(f"Experiment config file not found: {args.config_dir}")
    experiment_config = OmegaConf.load(args.config_dir)

    # Merge default config with experiment config
    config = OmegaConf.merge(default_config, experiment_config)

    # Apply overrides using the options argument
    if args.options:
        for kv in args.options:
            key, value = kv.split("=")
            # value is a string, try to convert it to the correct type
            if value in ['True', 'False']:
                value = True if value == 'True' else False
            elif value.isdigit():
                value = int(value)
            elif '.' in value and all([x.isdigit() for x in value.split('.')]):
                value = float(value)
            OmegaConf.update(config, key, value)

    return config


def main():
    ############################################################
    # Parse arguments and load configuration
    ############################################################

    config = parse_args()

    # Set up environment variables for distributed training
    config.distributed.local_rank, config.distributed.rank, config.distributed.world_size = world_info_from_env()
    device_id, is_distributed = init_distributed_device(config.distributed)

    # Setup logging
    output_path = os.path.join(config.experiment.output_dir, config.experiment.id)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'ckpts'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'results'), exist_ok=True)
    logger = setup_logger(log_file=os.path.join(output_path, f'{config.experiment.id}.log'), rank=config.distributed.rank)

    # Log configuration (from OmegaConf)
    logger.info(f'********************** Start logging **********************')
    # log_config_to_file(config, logger=logger)
    
    # Ramdom seed setting
    setup_seeds(seed=config.experiment.seed)

    ############################################################
    # Set up data loaders and agents
    ############################################################
    logger.info('********************** Setting up dataloaders **********************')

    environment = create_environment(logger)
    logger.info("Finshed building MP3D environment")

    val_envs = create_val_env(config, logger, environment, device_id)
    logger.info("Finshed building data batch env")

    ############################################################
    # Load agent 
    ############################################################
    



if __name__ == '__main__':
    main()

    # # Setup logging
    # import logging
    # logger = logging.getLogger(__name__)
    # logging.basicConfig(level=logging.INFO)

    # from PIL import Image
    # from models import get_models
    # model_cls = get_models("qwen2_5_vl", logger)
    # model = model_cls(config={})

    #  # Prepare the input data
    # image_paths = [
    #     "./data/0b22fa63d0f54a529c525afbf2e8bb25_0.png",
    #     "./data/0b22fa63d0f54a529c525afbf2e8bb25_1.png",
    #     "./data/0b22fa63d0f54a529c525afbf2e8bb25_2.png",
    #     "./data/0b22fa63d0f54a529c525afbf2e8bb25_3.png",
    # ]
    # images = []
    # for path in image_paths:
    #     try:
    #         img = Image.open(path).convert("RGB")
    #         images.append(img)
    #     except FileNotFoundError:
    #         logger.error(f"Image file not found: {path}")
    #         exit()

    # input_data = {
    #     "views": images,
    #     "instruction": "Describe the scene in detail."
    # }
    # prompt = "{image} {instruction}"

    # output = model(prompt, input_data)

    # print(output)

    # print("done")

    