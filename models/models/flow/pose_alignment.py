from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class AlignConfig:
    enable: bool = True
    use_rotation: bool = True
    use_scale: bool = True
    min_points: int = 2
    fallback: str = "translation"  # "translation" or "center"
    center_value: float = 0.5
    use_skeleton_weights: bool = False
    skeleton_weight_strength: float = 1.0
    init_from_heatmap: bool = True


def similarity_align(
    x0: torch.Tensor,
    x1: torch.Tensor,
    weights: torch.Tensor,
    use_rotation: bool = True,
    use_scale: bool = True,
    min_points: int = 2,
    eps: float = 1e-6,
    fallback: str = "translation",
    center_value: float = 0.5,
) -> torch.Tensor:
    """Align x0 to x1 with a weighted similarity transform.

    Args:
        x0/x1: [B, N, 2] normalized coordinates.
        weights: [B, N] non-negative weights (visibility or reliability).
    """
    if weights.dim() == 3:
        weights = weights.squeeze(-1)

    w = weights.clamp(min=0.0)
    sum_w = w.sum(dim=1, keepdim=True)
    denom = sum_w.clamp(min=1.0)

    mu0 = (x0 * w[..., None]).sum(dim=1) / denom
    mu1 = (x1 * w[..., None]).sum(dim=1) / denom

    X0 = x0 - mu0[:, None, :]
    X1 = x1 - mu1[:, None, :]

    X0w = X0 * w[..., None]
    C = torch.bmm(X0w.transpose(1, 2), X1)  # [B, 2, 2]

    b = x0.shape[0]
    R = torch.eye(2, device=x0.device, dtype=x0.dtype).unsqueeze(0).repeat(b, 1, 1)
    s = torch.ones(b, device=x0.device, dtype=x0.dtype)

    valid = sum_w.squeeze(-1) >= float(min_points)

    if use_rotation:
        U, S, Vh = torch.linalg.svd(C)
        R_candidate = torch.bmm(Vh.transpose(1, 2), U.transpose(1, 2))
        det = torch.det(R_candidate)
        if (det < 0).any():
            Vh = Vh.clone()
            Vh[det < 0, :, 1] *= -1.0
            R_candidate = torch.bmm(Vh.transpose(1, 2), U.transpose(1, 2))
        R = torch.where(valid[:, None, None], R_candidate, R)

        if use_scale:
            var0 = (w * (X0 ** 2).sum(dim=-1)).sum(dim=1)
            s_candidate = S.sum(dim=1) / (var0 + eps)
            s = torch.where(valid, s_candidate, s)
    else:
        if use_scale:
            var0 = (w * (X0 ** 2).sum(dim=-1)).sum(dim=1)
            var1 = (w * (X1 ** 2).sum(dim=-1)).sum(dim=1)
            s_candidate = torch.sqrt((var1 + eps) / (var0 + eps))
            s = torch.where(valid, s_candidate, s)

    t = mu1 - s[:, None] * torch.bmm(R, mu0.unsqueeze(-1)).squeeze(-1)

    x0_rot = torch.bmm(R, x0.transpose(1, 2)).transpose(1, 2)
    x0_align = s[:, None, None] * x0_rot + t[:, None, :]

    if fallback == "center":
        x0_center = x0 - mu0[:, None, :] + center_value
        x0_align = torch.where(valid[:, None, None], x0_align, x0_center)

    return x0_align
