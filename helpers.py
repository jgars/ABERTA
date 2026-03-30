import json
from pathlib import Path

import torch
import numpy as np


def get_device(show_info=True) -> torch.device:
    # If there is a GPU available, then use the GPU.
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        if show_info:
            print("GPU is available")
    else:
        device = torch.device("cpu")
        if show_info:
            print("GPU not available, CPU used")
    return device
