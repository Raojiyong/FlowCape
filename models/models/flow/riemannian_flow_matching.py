"""
Riemannian Flow Matching utilities for pose generation.

This module provides:
- Configuration dataclass for flow matching hyperparameters.
- Metric builder using graph Laplacian from skeleton edges.
- Interpolation and target velocity helpers in Riemannian settings
  (linearized with metric weighting).
- Loss computation for metric-weighted vector-field regression.
- Simple ODE integrator for inference with optional guidance
  (e.g., Weak-Shot keyness gradient) injected externally.

References followed for correctness:
1) Conditional Flow Matching template:
   https://github.com/facebookresearch/flow-matching
2) Flow ODE solve pattern used in Point-E:
   https://github.com/openai/point-e/blob/main/point_e/flows/flow_utils.py

The implementation here is self-contained and does not depend on
the rest of the training loop; it can be imported by detectors to
construct losses and run ODE integration at test time.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class RFMConfig:
    """Configuration for Riemannian Flow Matching.

    Attributes:
        num_timesteps: Number of integration steps during inference.
        t_epsilon: Lower bound to avoid t=0 instability.
        guidance_scale: Multiplier for external guidance term (e.g., keyness).
        atol: Absolute tolerance for adaptive solvers (unused in Euler).
        rtol: Relative tolerance for adaptive solvers (unused in Euler).
    """

    num_timesteps: int = 20
    t_epsilon: float = 1e-3
    guidance_scale: float = 0.0
    atol: float = 1e-5
    rtol: float = 1e-5


def laplacian_metric(
    skeleton: torch.Tensor, mask: torch.Tensor, num_points: int
) -> torch.Tensor:
    """Build a Laplacian-derived metric matrix per sample.

    Args:
        skeleton: Edge list of shape [B, E, 2] with 0-based indices.
        mask: Visibility mask of shape [B, N] (1=visible, 0=missing).
        num_points: Total number of keypoints (N).

    Returns:
        Metric tensor M of shape [B, N, N]; positive semi-definite.
    """

    device = skeleton.device
    batch_size = skeleton.shape[0]
    # Initialize adjacency
    adj = torch.zeros(batch_size, num_points, num_points, device=device)
    if skeleton.numel() > 0:
        src = skeleton[..., 0]
        dst = skeleton[..., 1]
        valid = (src >= 0) & (dst >= 0) & (src < num_points) & (dst < num_points)
        if valid.any():
            idx_b = torch.arange(batch_size, device=device)[:, None].expand_as(src)
            idx_b = idx_b[valid]
            src = src[valid]
            dst = dst[valid]
            adj[idx_b, src, dst] = 1.0
            adj[idx_b, dst, src] = 1.0

    # Degree matrix
    deg = adj.sum(dim=-1)
    lap = torch.diag_embed(deg) - adj

    # Mask out invisible points by zeroing corresponding rows/cols
    m = mask.float()
    lap = lap * m[:, None, :] * m[:, :, None]

    # Add small jitter to diagonal for stability
    lap = lap + torch.eye(num_points, device=device)[None] * 1e-4
    return lap


def geodesic_interp(
    x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Linearized geodesic interpolation between poses.

    Args:
        x0: Source pose [B, N, 2].
        x1: Target pose [B, N, 2].
        t: Time scalar or tensor broadcastable to [B, 1, 1].
    """

    return (1.0 - t) * x0 + t * x1


def target_velocity(
    x0: torch.Tensor, x1: torch.Tensor, metric: torch.Tensor = None
) -> torch.Tensor:
    """Compute target velocity field.

    For linear interpolation x_t = (1-t)x0 + t x1, the velocity is x1 - x0.
    The metric is not applied here; it is applied in the loss function.

    Args:
        x0: Source pose [B, N, 2].
        x1: Target pose [B, N, 2].
        metric: Unused (kept for API compatibility).
    """

    v = x1 - x0  # [B, N, 2]
    return v


def riemannian_flow_loss(
    v_pred: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    metric: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Metric-weighted L2 loss between predicted and target velocity fields.

    Args:
        v_pred: Predicted vector field [B, N, 2].
        x0/x1: Source and target poses [B, N, 2].
        metric: Metric matrix [B, N, N].
        t: Sampled time in [0,1] (broadcastable to [B,1,1]).
    """

    x_t = geodesic_interp(x0, x1, t)
    v_t = target_velocity(x0, x1, metric)
    diff = v_pred - v_t
    # Quadratic form per coordinate:
    # quad = torch.einsum("bij,bjk,bik->bi", diff, metric, diff)
    # For each coordinate d in {x,y}, compute diff_d^T * M * diff_d.
    # diff: [B, N, 2], metric: [B, N, N]
    mdiff = torch.einsum("bnm,bmd->bnd", metric, diff)  # [B, N, 2]
    quad = (diff * mdiff).sum(dim=1)  # [B, 2]
    loss = quad.sum(dim=-1).mean()
    return loss, x_t


class FlowODEIntegrator:
    """Simple Euler ODE integrator for flow matching inference.

    External guidance can be injected via a callable that returns
    a tensor shaped like the state (e.g., keyness gradient).
    """

    def __init__(self, config: RFMConfig):
        self.cfg = config

    def integrate(
        self,
        x0: torch.Tensor,
        vector_field: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        guidance: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Integrate dx/dt = v(x,t) from t_eps to 1.

        Args:
            x0: Initial pose [B, N, 2].
            vector_field: Callable (x, t) -> v [B, N, 2].
            guidance: Optional callable (x, t) -> g [B, N, 2].
        Returns:
            x_T: Pose at t=1.
        """

        x = x0
        steps = self.cfg.num_timesteps
        dt = (1.0 - self.cfg.t_epsilon) / steps
        t = torch.full((x.shape[0], 1, 1), self.cfg.t_epsilon, device=x.device)

        for _ in range(steps):
            v = vector_field(x, t)
            if guidance is not None and self.cfg.guidance_scale > 0:
                g = guidance(x, t)
                v = v + self.cfg.guidance_scale * g
            x = x + dt * v
            t = t + dt

        return x


def soft_argmax_heatmap(
    heatmap: torch.Tensor, temperature: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert heatmaps to coords and confidences with soft-argmax.

    Args:
        heatmap: [B, N, H, W]
        temperature: Softmax temperature.

    Returns:
        coords: [B, N, 2] normalized to [0,1] (x, y).
        conf: [B, N] confidence (max probability).
    """

    b, n, h, w = heatmap.shape
    logits = heatmap.view(b, n, -1) / temperature
    probs = torch.softmax(logits, dim=-1)

    # Coordinate grid
    ys = torch.linspace(0, 1, h, device=heatmap.device)
    xs = torch.linspace(0, 1, w, device=heatmap.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # [H*W,2]

    coords = torch.einsum("bnp,pd->bnd", probs, grid)
    conf = probs.max(dim=-1).values
    return coords, conf
