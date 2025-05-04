import json
import copy
import torch
import numpy as np
import cv2
from typing import Any, List, Mapping, Optional

import sys
# sys.path.append('/home/rongtao/gengze/NavGPT-2/nav_src/LLMs')
from .emu_models.emu_model import Emu

from accelerate import init_empty_weights
from peft import LoraConfig, get_peft_model
from accelerate import load_checkpoint_and_dispatch
from accelerate import infer_auto_device_map

def prepare_model(args):

    model_type = args.llm_model_name
    instruct = args.instruct
    ckpt_path = args.ckpt_path

    with open(f'LLMs/emu_models/{model_type}.json', "r", encoding="utf8") as f:
        model_cfg = json.load(f)
    print(f"=====> model_cfg: {model_cfg}")

    # with init_empty_weights():

    model = Emu(**model_cfg, cast_dtype=torch.float, args=args)

    if instruct:
        print('Patching LoRA...')
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.decoder.lm = get_peft_model(model.decoder.lm, lora_config)

    print(f"=====> Loading from ckpt_path {ckpt_path}")
    my_device_map = {
        'visual': 0, 
        'ln_visual': 0, 
        'vl_adapter': 0,
        'decoder.lm.base_model.model.model.embed_tokens': 0,
        'decoder.lm.base_model.model.model.layers.0': 0,
        'decoder.lm.base_model.model.model.layers.1': 0,
        'decoder.lm.base_model.model.model.layers.2': 0,
        'decoder.lm.base_model.model.model.layers.3': 0,
        'decoder.lm.base_model.model.model.layers.4': 0,
        'decoder.lm.base_model.model.model.layers.5': 0,
        'decoder.lm.base_model.model.model.layers.6': 0,
        'decoder.lm.base_model.model.model.layers.7': 0,
        'decoder.lm.base_model.model.model.layers.8': 0,
        'decoder.lm.base_model.model.model.layers.9': 0,
        'decoder.lm.base_model.model.model.layers.10': 0,
        'decoder.lm.base_model.model.model.layers.11': 0,
        'decoder.lm.base_model.model.model.layers.12': 0,
        'decoder.lm.base_model.model.model.layers.13': 0,
        'decoder.lm.base_model.model.model.layers.14': 0,
        'decoder.lm.base_model.model.model.layers.15': 0,
        'decoder.lm.base_model.model.model.layers.16': 0,
        'decoder.lm.base_model.model.model.layers.17': 0,
        'decoder.lm.base_model.model.model.layers.18': 0,
        'decoder.lm.base_model.model.model.layers.19': 0,
        'decoder.lm.base_model.model.model.layers.20': 1,
        'decoder.lm.base_model.model.model.layers.21': 1,
        'decoder.lm.base_model.model.model.layers.22': 1,
        'decoder.lm.base_model.model.model.layers.23': 1,
        'decoder.lm.base_model.model.model.layers.24': 1,
        'decoder.lm.base_model.model.model.layers.25': 1,
        'decoder.lm.base_model.model.model.layers.26': 1,
        'decoder.lm.base_model.model.model.layers.27': 1,
        'decoder.lm.base_model.model.model.layers.28': 1,
        'decoder.lm.base_model.model.model.layers.29': 1,
        'decoder.lm.base_model.model.model.layers.30': 1,
        'decoder.lm.base_model.model.model.layers.31': 1,
        'decoder.lm.base_model.model.model.layers.32': 1,
        'decoder.lm.base_model.model.model.layers.33': 1,
        'decoder.lm.base_model.model.model.layers.34': 1,
        'decoder.lm.base_model.model.model.layers.35': 1,
        'decoder.lm.base_model.model.model.layers.36': 1,
        'decoder.lm.base_model.model.model.layers.37': 1,
        'decoder.lm.base_model.model.model.layers.38': 1,
        'decoder.lm.base_model.model.model.layers.39': 1,
        'decoder.lm.base_model.model.model.norm': 1,
        'decoder.lm.base_model.model.lm_head': 1,
        'decoder.lm.base_model.model.stu_regress_head': 1}
    # my_device_map = infer_auto_device_map(model, max_memory={0: "20GiB", 1: "20GiB"}, no_split_module_classes=['visual', 'LlamaDecoderLayer', 'T5Block'])
    # model = load_checkpoint_and_dispatch(model, checkpoint=ckpt_path, device_map=my_device_map, no_split_module_classes=['visual', 'LlamaDecoderLayer', 'T5Block'])
    model = load_checkpoint_and_dispatch(model, checkpoint=ckpt_path, device_map=my_device_map)
    # ckpt = torch.load(ckpt_path, map_location="cpu")
    # msg = model.load_state_dict(ckpt, strict=False)
    model.to(torch.bfloat16).eval()
    print(f"Loaded successfully")

    return model
import math
class Custom_Emu(object):
    '''Custom Emu model for NavGPT-2'''
    def __init__(
        self,
        config: Any,
        max_new_tokens: int = 512,
        num_beams: int = 5,
        length_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
    ):
        self.model_type = config.llm_model_name
        self.instruct = config.instruct
        self.ckpt_path = config.ckpt_path
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty

        # Create a dummy args
        args = copy.deepcopy(config)
        args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = prepare_model(args)

    @property
    def _llm_type(self) -> str:
        return "custom_Emu"

    def _process_img(self, img, device):
        width, height = 224, 224
        OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
        OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
        # Resize the image
        img = cv2.resize(img, (width, height))
        # Normalize image pixel values to [0,1]
        img = np.array(img) / 255.
        # Standardize using the mean and std values
        img = (img - OPENAI_DATASET_MEAN) / OPENAI_DATASET_STD
        # Convert the numpy array to a PyTorch tensor
        img = torch.tensor(img).to(device).to(torch.float)
        # Rearrange from HWC to CHW format
        img = torch.einsum('hwc->chw', img)
        # Add a batch dimension
        img = img.unsqueeze(0)

        return img


    def _xy(self,data):
        heading = data["heading"]
        distance = data["distance"]
        x = round(distance * math.sin(heading), 4)
        y = round(distance * math.cos(heading), 4)
        return [x, y]

    def _construct_input(self, input, prompt):
        # image processing
        images = []
        for i in range(len(input["views"])):
            img = input["views"][i]
            img = self._process_img(img, self.model.args.device)
            images.append(img)
        images = torch.cat(images, dim=0)

        # text processing
        image_p = '[IMG]<image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image>[/IMG]'

        prompt = prompt.replace("{instruction}", input['instruction'])
        prompt = prompt.replace("{cur_image}", image_p)
        #candidate_str = str({key: [round(np.rad2deg(val['heading']), 2), round(np.rad2deg(val['elevation']), 2)] for key, val in input['candidate'].items()})
        candidate_str = str(
            {key: self._xy(val) for key, val in input['candidate'].items()})
        prompt = prompt.replace("{candidates}", candidate_str)
        prompt = prompt.replace("{history}", str(input['history']))
        
        if input["map"] is not None:
            pass
        full_input = prompt + " [ASSISTANT]:"
        return images, full_input

    def __call__(
        self,
        prompt: str,
        input: dict,
    ) -> str:

        images, full_input = self._construct_input(input, prompt)

        samples = {"image": images, "prompt": full_input}
        result = self.model.generate(
            samples,
            max_new_tokens = self.max_new_tokens,
            num_beams = self.num_beams,
            length_penalty = self.length_penalty,
            repetition_penalty = self.repetition_penalty,
        )[0].strip()
        return result

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_type": self.model_type,
            "instruct": self.instruct,
            "ckpt_path": self.ckpt_path,
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "length_penalty": self.length_penalty,
            "repetition_penalty": self.repetition_penalty,
            }