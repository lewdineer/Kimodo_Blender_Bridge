import numpy as np, torch, random
def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
def to_numpy(d):
    return {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in d.items()}
