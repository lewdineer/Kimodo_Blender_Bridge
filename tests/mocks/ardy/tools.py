import random

import numpy as np
import torch


def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def to_numpy(obj):
    """Mirror ardy.tools.to_numpy, which recurses rather than assuming a dict.

    Kept faithful on purpose: an earlier dict-only version made a bare tensor
    raise here, which looks exactly like a bridge bug and is not one.
    """
    if isinstance(obj, dict):
        return {k: to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_numpy(x) for x in obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()
    return obj
