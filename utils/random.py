import os

import torch
import random
import numpy as np

def fix_random_seeds(seed: int = 31) -> None:
    """
    Fix random seeds.
    Copy from: https://github.com/The-AI-Summer/self-attention-cv/blob/8280009366b633921342db6cab08da17b46fdf1c/self_attention_cv/common.py#L16
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
