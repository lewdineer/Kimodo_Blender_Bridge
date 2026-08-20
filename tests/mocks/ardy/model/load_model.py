"""Mock ARDY model reproducing the real call contract (shapes + asserts)."""
import json, math, os, torch
from ardy.skeleton import CoreSkeleton27

DEFAULT_MODEL = "core"


def _record(kind, payload):
    """Append one event to the log the test process reads."""
    path = os.environ.get("ARDY_MOCK_LOG")
    if not path:
        return
    with open(path, "a") as fh:
        fh.write(json.dumps({"kind": kind, **payload}) + "\n")


class _MotionRep:
    def __init__(self, skel, fps):
        self.skeleton = skel
        self.fps = fps
        self.J = skel.nbjoints
        self.D = 3 + self.J * 9

    def inverse(self, motion, is_normalized=True):
        B, T, D = motion.shape
        assert D == self.D
        root = motion[:, :, :3]
        rot = motion[:, :, 3:].reshape(B, T, self.J, 3, 3)
        # re-orthonormalize so the result is a valid rotation
        u, _, vh = torch.linalg.svd(rot.reshape(-1, 3, 3))
        rot = (u @ vh).reshape(B, T, self.J, 3, 3)
        return {
            "local_rot_mats": rot,
            "root_positions": root,
            "posed_joints": torch.zeros(B, T, self.J, 3),
            "global_rot_mats": rot.clone(),
            "foot_contacts": torch.zeros(B, T, 4),
        }

    def create_conditions_from_constraints_batched(self, lst, lengths, to_normalize, device):
        T = int(lengths.max()); B = len(lengths)
        obs = torch.zeros(B, T, self.D, device=device)
        mask = torch.zeros(B, T, self.D, device=device)
        for c in lst:
            for f in c.frame_indices.tolist():
                if 0 <= f < T:
                    mask[:, f, :3] = 1.0
                    obs[:, f, :3] = 1.0
        return obs, mask


class _Model:
    def __init__(self, name, device):
        self.name = name
        self.device = device
        self.skeleton = CoreSkeleton27()
        self.motion_rep = _MotionRep(self.skeleton, fps=20.0)
        self.num_frames_per_token = 4
        self.gen_horizon_len = 40
        self.diffusion = type("D", (), {"num_base_steps": 32})()

    def __call__(self, texts, num_frames, num_denoising_steps=None, pad_mask=None,
                 first_heading_angle=None, motion_mask=None, observed_motion=None,
                 cfg_weight=None, crop_history_length=None,
                 init_history_sequence=None, **kw):
        # Mirror the real model's contract so the caller is genuinely validated.
        assert isinstance(texts, list) and len(texts) == 1, texts
        assert pad_mask is not None and pad_mask.shape[1] == num_frames
        assert 1 <= num_denoising_steps <= self.diffusion.num_base_steps
        if init_history_sequence is not None:
            assert first_heading_angle is None, "first_heading_angle must be None with init history"
            assert crop_history_length is None, "not supporting both crop_history_length and init_history_sequence"
            assert init_history_sequence.shape[1] % self.num_frames_per_token == 0
        else:
            assert first_heading_angle is not None
            if crop_history_length is not None:
                assert crop_history_length % self.num_frames_per_token == 0
        if motion_mask is not None:
            assert motion_mask.shape[1] == num_frames, \
                f"motion_mask len {motion_mask.shape[1]} != num_frames {num_frames}"
            assert observed_motion.shape[1] == num_frames
        _record("call", dict(text=texts[0], num_frames=num_frames,
                             hist=0 if init_history_sequence is None else int(init_history_sequence.shape[1]),
                             steps=num_denoising_steps, cfg=list(cfg_weight),
                             crop=crop_history_length,
                             constrained=motion_mask is not None))

        init_len = 0 if init_history_sequence is None else int(init_history_sequence.shape[1])
        steps = math.ceil((num_frames - init_len) / self.gen_horizon_len)
        total = steps * self.gen_horizon_len + init_len      # >= num_frames

        out = torch.zeros(1, total, self.motion_rep.D)
        if init_len:
            out[:, :init_len] = init_history_sequence
        eye = torch.eye(3).reshape(-1).repeat(self.motion_rep.J)
        for t in range(init_len, total):
            out[0, t, :3] = torch.tensor([0.01 * t, 0.9, 0.0])
            out[0, t, 3:] = eye
        return out


def load_model(modelname=None, device=None, eval_mode=True, default_family=None,
               text_encoder=None, text_encoder_fp32=False, text_encoder_mode=None,
               text_encoder_url=None, return_config=False, checkpoints_dir=None):
    assert text_encoder_mode == "local", f"expected local encoder, got {text_encoder_mode!r}"
    return _Model(modelname, device)
