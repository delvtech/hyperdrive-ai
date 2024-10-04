import random

import numpy as np
import torch


def set_seeds(seed: int | None = None):
    seed = seed if seed is not None else np.random.randint(1, 99999)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Global Seed: {seed}")
