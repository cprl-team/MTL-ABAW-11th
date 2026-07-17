import os
import random

import numpy as np


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy, and (if present) PyTorch. Call once at run start."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
