"""Mock of the rotation helpers ardy_bridge imports from ardy.geometry."""
import torch


def matrix_to_axis_angle(R):
    """Rotation matrices [..., 3, 3] -> axis-angle vectors [..., 3].

    Real implementation, not a stub: the bridge hands these straight to Blender
    as pose rotations, so a placeholder would let a wrong conversion through.
    """
    shape = R.shape[:-2]
    m = R.reshape(-1, 3, 3)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cos)

    axis = torch.stack([
        m[:, 2, 1] - m[:, 1, 2],
        m[:, 0, 2] - m[:, 2, 0],
        m[:, 1, 0] - m[:, 0, 1],
    ], dim=-1)
    sin = torch.sin(angle).unsqueeze(-1)
    small = sin.abs() < 1e-8
    axis = torch.where(small, torch.zeros_like(axis), axis / (2.0 * sin.clamp(min=1e-12)))
    return (axis * angle.unsqueeze(-1)).reshape(*shape, 3)
