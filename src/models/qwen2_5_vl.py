import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchvision import transforms
import numpy as np

from src.models.base_mllm import BaseMLLM

class Qwen2_5_VL(BaseMLLM):
    pass