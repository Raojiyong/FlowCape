"""
Graph-aware transformer blocks for vector-field prediction on poses.

Features:
- Time embedding (sinusoidal) injected into node features.
- Graph Laplacian smoothing (GCN-style) over skeleton adjacency.
- Metric construction from skeleton + visibility to reuse in flow losses.
- Geodesic (linearized) interpolation helpers for RFM-style training.

References for correctness:
1) Conditional flow matching module design:
   https://github.com/facebookresearch/flow-matching
2) Timestep embedding pattern:
   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py
3) Laplacian smoothing idea (GCN):
   https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/conv/gcn_conv.py
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.models.flow.riemannian_flow_matching import laplacian_metric


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal time embeddings."""
    half = dim // 2
    freq = torch.exp(
        torch.arange(half, device=t.device, dtype=torch.float32)
        * -(torch.log(torch.tensor(10000.0)) / (half - 1))
    )
    args = t * freq[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def adjacency_from_edges(batch_edges: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Build adjacency matrix from edge list.

    Args:
        batch_edges: [B, E, 2] long tensor.
        num_nodes: number of nodes (N).
    Returns:
        adj: [B, N, N] symmetric adjacency.
    """
    b = batch_edges.shape[0]
    adj = torch.zeros((b, num_nodes, num_nodes), device=batch_edges.device)
    if batch_edges.numel() == 0:
        return adj

    src = batch_edges[..., 0]
    dst = batch_edges[..., 1]
    valid = (src >= 0) & (dst >= 0) & (src < num_nodes) & (dst < num_nodes)
    if not valid.any():
        return adj

    idx_b = torch.arange(b, device=batch_edges.device)[:, None].expand_as(src)
    idx_b = idx_b[valid]
    src = src[valid]
    dst = dst[valid]
    adj[idx_b, src, dst] = 1.0
    adj[idx_b, dst, src] = 1.0
    return adj


def geodesic_interp(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linearized geodesic interpolation (Exp/Log approximated by linear blend)."""
    return (1.0 - t) * x0 + t * x1


class LaplacianGCN(nn.Module):
    """Single-step Laplacian smoothing."""

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C], adj: [B, N, N]
        deg = adj.sum(dim=-1, keepdim=True) + 1e-6
        return torch.bmm(adj, x) / deg


class AdaLN(nn.Module):
    """Adaptive Layer Norm modulated by conditioning (idea.md Step 4).
    
    Text embeddings modulate the normalization via learned scale/shift:
    AdaLN(x, cond) = (1 + gamma(cond)) * LayerNorm(x) + beta(cond)
    """
    
    def __init__(self, dim: int, cond_dim: int = None):
        super().__init__()
        cond_dim = cond_dim or dim
        self.norm = nn.LayerNorm(dim)
        self.gamma_proj = nn.Linear(cond_dim, dim)
        self.beta_proj = nn.Linear(cond_dim, dim)
        
        # Initialize to identity
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim] features to normalize
            cond: [B, dim] or [B, N, dim] conditioning (e.g., text embedding)
        """
        x = self.norm(x)
        
        # Handle different conditioning shapes
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)  # [B, 1, dim]
        
        gamma = self.gamma_proj(cond)  # [B, 1, dim] or [B, N, dim]
        beta = self.beta_proj(cond)
        
        return x * (1 + gamma) + beta


class GraphFlowBlock(nn.Module):
    """Graph-aware transformer block for node feature refinement with AdaLN."""

    def __init__(self, dim: int, nhead: int, dropout: float = 0.1, use_adaln: bool = True):
        super().__init__()
        self.use_adaln = use_adaln
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        
        # Use AdaLN for text-modulated normalization
        if use_adaln:
            self.norm1 = AdaLN(dim)
            self.norm2 = AdaLN(dim)
            self.norm3 = AdaLN(dim)
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.norm3 = nn.LayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
        self.gcn = LaplacianGCN()

    def forward(
        self,
        node_feat: torch.Tensor,
        memory: torch.Tensor,
        adj: torch.Tensor,
        text_cond: torch.Tensor = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_feat: [B, N, C] node features
            memory: [B, M, C] image memory
            adj: [B, N, N] adjacency matrix
            text_cond: [B, C] text conditioning for AdaLN (optional)
            key_padding_mask: [B, N] True indicates padded nodes
        """
        # Self-attention over nodes
        residual = node_feat
        node_feat = self.self_attn(
            node_feat,
            node_feat,
            node_feat,
            key_padding_mask=key_padding_mask)[0]
        node_feat = self.dropout(node_feat) + residual
        if self.use_adaln and text_cond is not None:
            node_feat = self.norm1(node_feat, text_cond)
        else:
            node_feat = self.norm1(node_feat) if not self.use_adaln else self.norm1(node_feat, node_feat.mean(dim=1))

        # Cross-attention to image memory
        residual = node_feat
        node_feat = self.cross_attn(node_feat, memory, memory)[0]
        node_feat = self.dropout(node_feat) + residual
        if self.use_adaln and text_cond is not None:
            node_feat = self.norm2(node_feat, text_cond)
        else:
            node_feat = self.norm2(node_feat) if not self.use_adaln else self.norm2(node_feat, node_feat.mean(dim=1))

        # Graph smoothing
        smoothed = self.gcn(node_feat, adj)

        # FFN
        residual = smoothed
        node_feat = self.ffn(smoothed)
        node_feat = self.dropout(node_feat) + residual
        if self.use_adaln and text_cond is not None:
            node_feat = self.norm3(node_feat, text_cond)
        else:
            node_feat = self.norm3(node_feat) if not self.use_adaln else self.norm3(node_feat, node_feat.mean(dim=1))
        
        return node_feat


@dataclass
class GraphFlowConfig:
    dim: int = 256
    nhead: int = 8
    num_layers: int = 3
    time_embed_dim: int = 128
    dropout: float = 0.1


class GraphFlowEncoderDecoder(nn.Module):
    """Encoder-decoder for vector-field prediction with graph smoothing."""

    def __init__(self, cfg: GraphFlowConfig):
        super().__init__()
        self.cfg = cfg
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.dim),
            nn.SiLU(),
            nn.Linear(cfg.dim, cfg.dim),
        )
        self.blocks = nn.ModuleList(
            [GraphFlowBlock(dim=cfg.dim, nhead=cfg.nhead, dropout=cfg.dropout, use_adaln=True) for _ in range(cfg.num_layers)]
        )
        self.vector_head = nn.Linear(cfg.dim, 2)

    def forward(
        self,
        node_feat: torch.Tensor,
        memory: torch.Tensor,
        skeleton: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        text_cond: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_feat: [B, N, C] (text-conditioned tokens).
            memory: [B, M, C] (flattened image features).
            skeleton: [B, E, 2] edge indices.
            mask: [B, N, 1] visibility mask.
            t: [B, 1, 1] timestep.
            text_cond: [B, C] text conditioning for AdaLN (optional, idea.md Step 4).
        Returns:
            coords_pred: [B, N, 2] predicted coordinates in [0,1]
            metric: [B, N, N]
        """
        b, n, _ = node_feat.shape
        num_points = n

        mask_f = mask.float()
        mask_bool = mask_f.squeeze(-1) > 0

        t_emb = sinusoidal_time_embedding(t.view(b, 1), dim=self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)[:, None, :]  # [B,1,C]
        node_feat = (node_feat + t_emb) * mask_f

        adj = adjacency_from_edges(skeleton, num_points)
        adj = adj * mask_bool[:, None, :] * mask_bool[:, :, None]
        
        # Use mean of node features as text conditioning if not provided
        if text_cond is None:
            denom = mask_f.sum(dim=1).clamp(min=1.0)
            text_cond = (node_feat * mask_f).sum(dim=1) / denom  # [B, C]

        pad_mask = ~mask_bool
        if pad_mask.any():
            all_true = pad_mask.all(dim=1)
            if all_true.any():
                pad_mask = pad_mask.clone()
                pad_mask[all_true, 0] = False

        for blk in self.blocks:
            node_feat = blk(node_feat, memory, adj, text_cond=text_cond, key_padding_mask=pad_mask)
            node_feat = node_feat * mask_f

        # Output velocity v_pred (linear, no sigmoid) for flow matching
        # Range can be positive or negative
        v_pred = self.vector_head(node_feat) * mask
        metric = laplacian_metric(skeleton, mask.squeeze(-1) > 0, num_points)
        return v_pred, metric
