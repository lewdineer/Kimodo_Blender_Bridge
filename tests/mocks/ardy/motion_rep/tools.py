import torch
def length_to_mask(lengths):
    m = int(lengths.max())
    return torch.arange(m, device=lengths.device)[None, :] < lengths[:, None]
