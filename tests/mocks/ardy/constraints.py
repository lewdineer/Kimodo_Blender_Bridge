import torch

TYPES = {"root2d", "fullbody", "left-hand", "right-hand", "left-foot", "right-foot", "end-effector"}


class C:
    def __init__(self, d):
        self.type = d["type"]
        self.frame_indices = torch.tensor(d["frame_indices"])
        self.global_root_heading = None
        if d.get("global_root_heading") is not None:
            # Stored as handed over, exactly like Root2DConstraintSet.from_dict:
            # torch.tensor(...) accepts [N, 2] happily. The shape only becomes a
            # problem later, during conditioning -- see _MotionRep in the model
            # mock, which is where the real failure surfaces too.
            self.global_root_heading = torch.tensor(
                d["global_root_heading"], dtype=torch.float32)


def load_constraints_lst(path_or_data, skeleton):
    data = path_or_data
    out = []
    for el in data:
        if el["type"] not in TYPES:
            raise KeyError(el["type"])
        out.append(C(el))
    return out
