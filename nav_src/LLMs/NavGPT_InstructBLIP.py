import os
import sys
import copy
import torch
import numpy as np
from typing import Any, List, Mapping, Optional

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

sys.path.append(os.path.dirname(__file__))
from lavis.models import load_model

class NavGPTInstruct(object):
    '''Custom Emu model for NavGPT-2'''
    def __init__(
        self,
        config: Any,
        num_beams: int = 5,
        max_length: int = 512,
        min_length: int = 1,
        repetition_penalty: float = 1.0,
    ):
        self.model_type = config.llm_model_name          # "InstructBLIPNav-FlanT5XL"
        self.ckpt_path = config.ckpt_path
        self.num_beams = num_beams
        self.max_length = max_length
        self.min_length = min_length
        self.repetition_penalty = repetition_penalty

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = load_model(
            name='blip2_t5_instruct_nav',
            model_type='flant5xl',
            checkpoint=self.ckpt_path,
            is_eval=True,
            device= self.device,
        )

        OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
        OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    @property
    def _llm_type(self) -> str:
        return "NavGPT_InstructBLIP_FlanT5"

    def _xy(self,data):
        heading = data["heading"]
        distance = data["distance"]
        x = round(distance * np.sin(heading), 4)
        y = round(distance * np.cos(heading), 4)
        return [x, y]

    def _construct_input(self, input, prompt):
        # image processing
        images = []
        for i in range(len(input["views"])):
            img = input["views"][i]
            img = self.transform(img)
            images.append(img)

        # Get current heading
        heading = input["heading"]
        if heading < 0:
            heading = heading + 360
        orientation_idx = int((heading + 45) / 90)
        if orientation_idx > 3:
            orientation_idx = 0

        # Order iamges
        images = images[orientation_idx:] + images[:orientation_idx]
        images = torch.stack(images).to(self.device)

        # get candidate at different orientations
        front, left, right, back = [], [], [], []
        # change angle from redius to degree
        for key, value in input["candidate"].items():
            candidate_heading = np.rad2deg(value["heading"])
            candidate_elevation = np.rad2deg(value["elevation"])
            if candidate_heading < 0 or candidate_heading < 45:
                front.append(round(candidate_heading, 2))
            elif candidate_heading > 45 and candidate_heading < 135:
                right.append(round(candidate_heading, 2))
            elif candidate_heading > 135 and candidate_heading < 225:
                back.append(round(candidate_heading, 2))
            elif candidate_heading > 225 and candidate_heading < 315:
                left.append(round(candidate_heading, 2))
        
        candidate_list = [front, right, back, left]
        # Order candidates
        candidate_list = candidate_list[orientation_idx:] + candidate_list[:orientation_idx]
        input["candidate"] = candidate_list

        # text processing
        image_p = '[IMG]<image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image>[/IMG]'

        prompt = prompt.replace("{instruction}", input['instruction'])
        prompt = prompt.replace("{image}", image_p)
        # candidate_str = str({key: [round(np.rad2deg(val['heading']), 2), round(np.rad2deg(val['elevation']), 2)] for key, val in input['candidate'].items()})
        # candidate_str = str(
        #     {key: self._xy(val) for key, val in input['candidate'].items()})
        # prompt = prompt.replace("{candidates}", candidate_str)
        prompt = prompt.replace("{candidate_front}", str(input['candidate'][0]))
        prompt = prompt.replace("{candidate_right}", str(input['candidate'][1]))
        prompt = prompt.replace("{candidate_rear}", str(input['candidate'][2]))
        prompt = prompt.replace("{candidate_left}", str(input['candidate'][3]))
        prompt = prompt.replace("{history}", str(input['history']))
        
        sample = {
            "text_input": prompt,
            "qformer_text_input": [input["instruction"] for _ in range(4)],
            "images": images,
        }

        return sample

    def __call__(
        self,
        prompt: str,
        input: dict,
    ) -> str:

        samples = self._construct_input(input, prompt)

        with torch.cuda.amp.autocast(enabled=True):
            llm_thoughts = self.model.generate(
                samples,
                use_nucleus_sampling=False,
                num_beams=self.num_beams,
                max_length=self.max_length,
                min_length=self.min_length,
            )
        return llm_thoughts[0]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_type": self.model_type,
            "ckpt_path": self.ckpt_path,
            "num_beams": self.num_beams,
            "max_length": self.max_length,
            "min_length": self.min_length,
            "repetition_penalty": self.repetition_penalty,
            }