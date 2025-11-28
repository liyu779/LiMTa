
from typing import Any, Dict, List, Sequence, Tuple
import random
import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

class StrideEmbed(nn.Module):
    '''
    stride embedding layer
    '''
    def __init__(self, img_height=40, img_width=40, stride_size=4, in_chans=1, embed_dim=192):
        super().__init__()
        self.num_patches = img_height * img_width // stride_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=stride_size, stride=stride_size)

    def forward(self, x):
        x = self.proj(x).transpose(1, 2)
        return x