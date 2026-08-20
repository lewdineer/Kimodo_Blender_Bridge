import numpy as np, torch

CORE27 = [("Hips",None),("Spine","Hips"),("Spine1","Spine"),("Spine2","Spine1"),
 ("Spine3","Spine2"),("Neck","Spine3"),("Head","Neck"),("RightShoulder","Spine3"),
 ("RightArm","RightShoulder"),("RightForeArm","RightArm"),("RightHand","RightForeArm"),
 ("RightHandEnd","RightHand"),("RightHandThumb1","RightHand"),("LeftShoulder","Spine3"),
 ("LeftArm","LeftShoulder"),("LeftForeArm","LeftArm"),("LeftHand","LeftForeArm"),
 ("LeftHandEnd","LeftHand"),("LeftHandThumb1","LeftHand"),("RightUpLeg","Hips"),
 ("RightLeg","RightUpLeg"),("RightFoot","RightLeg"),("RightToeBase","RightFoot"),
 ("LeftUpLeg","Hips"),("LeftLeg","LeftUpLeg"),("LeftFoot","LeftLeg"),("LeftToeBase","LeftFoot")]

class CoreSkeleton27:
    name = "cskel27"
    def __init__(self):
        self.bone_order_names = [n for n, _ in CORE27]
        idx = {n: i for i, n in enumerate(self.bone_order_names)}
        self.joint_parents = torch.tensor([-1 if p is None else idx[p] for _, p in CORE27])
        self.root_idx = 0
        rng = np.random.default_rng(7)
        offs = rng.normal(0, 0.15, (len(CORE27), 3))
        n = np.zeros_like(offs)
        for i, p in enumerate(self.joint_parents.tolist()):
            n[i] = offs[i] if p < 0 else n[p] + offs[i]
        n[0] = 0.0
        self.neutral_joints = torch.tensor(n, dtype=torch.float32)
        self.nbjoints = len(CORE27)

class SOMASkeleton30:            # only used for an isinstance() check
    pass
