import torch
TYPES = {"root2d", "fullbody", "left-hand", "right-hand", "left-foot", "right-foot", "end-effector"}
class C:
    def __init__(self, d):
        self.type = d["type"]
        self.frame_indices = torch.tensor(d["frame_indices"])
def load_constraints_lst(path_or_data, skeleton):
    data = path_or_data
    out = []
    for el in data:
        if el["type"] not in TYPES:
            raise KeyError(el["type"])
        out.append(C(el))
    return out
