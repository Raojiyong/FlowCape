"""
Riemannian vector-field head for flow matching over poses.

Features:
- Time embedding (sinusoidal) injected into node features.
- Laplacian-aware smoothing over skeleton adjacency.
- Predicts a vector field v(x_t, t, cond) instead of absolute coords.
- Metric-weighted flow loss using the RFM utilities.
- Optional heatmap prediction branch for compatibility.

References:
1) Conditional flow matching vector-field structure:
   https://github.com/facebookresearch/flow-matching
2) Timestep embedding pattern:
   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmpose.core.evaluation import keypoint_pck_accuracy
from mmpose.core.post_processing import transform_preds
from mmpose.models import HEADS

from models.models.flow.riemannian_flow_matching import (
    laplacian_metric,
    riemannian_flow_loss,
)
from models.models.utils.graph_flow_encoder import (
    GraphFlowConfig,
    GraphFlowEncoderDecoder,
    sinusoidal_time_embedding,
)


@dataclass
class RiemannianHeadConfig:
    in_channels: int
    text_channels: int
    hidden_dim: int = 256
    time_embed_dim: int = 128
    with_heatmap: bool = True
    heatmap_size: Optional[int] = 64
    temperature: float = 0.01
    dropout: float = 0.1


@HEADS.register_module()
class RiemannianPoseHead(nn.Module):
    """Riemannian vector-field head that predicts v(x, t)."""

    def __init__(self,
                 img_in_channels,
                 text_in_channels,
                 hidden_dim=256,
                 time_embed_dim=128,
                 with_heatmap=True,
                 heatmap_size=64,
                 dropout=0.1,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.with_heatmap = with_heatmap
        self.heatmap_size = heatmap_size
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

        self.img_proj = nn.Conv2d(img_in_channels, hidden_dim, kernel_size=1)
        self.text_proj = nn.Linear(text_in_channels, hidden_dim)
        self.coord_proj = nn.Linear(2, hidden_dim)  # Project coords (x_t) to hidden dim
        self.time_embed_dim = time_embed_dim
        self.dropout = nn.Dropout(dropout)

        self.vector_head = GraphFlowEncoderDecoder(
            GraphFlowConfig(
                dim=hidden_dim,
                nhead=8,
                num_layers=3,
                time_embed_dim=time_embed_dim,
                dropout=dropout))

        if self.with_heatmap:
            self.heatmap_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, feature_s, target_s, mask_s, skeleton, point_descriptions, t=None, coords=None):
        """
        Maintains signature compatibility with PoseHead for detector wiring.

        Args:
            x: [B, C, H, W] query features.
            feature_s, target_s: unused placeholders for compatibility.
            mask_s: [B, N, 1] visibility.
            skeleton: list of edge lists length B.
            point_descriptions: [B, N, Ctxt] text embeddings.
            t: optional [B,1,1] timestep; if None random U[0,1].
            coords: optional [B,N,2] coords; if None zeros.
        Returns:
            output: stacked v_pred [1, B, N, 2]
            initial_proposals: coords input (or zeros) [B, N, 2]
            similarity_map: None (placeholder)
        """
        b, _, h, w = x.shape
        feat = self.img_proj(x)  # [B, hidden, H, W]
        memory = feat.flatten(2).transpose(1, 2)  # [B, HW, hidden]

        txt = self.text_proj(point_descriptions)  # [B, N, hidden]

        if t is None:
            t = torch.rand((b, 1, 1), device=x.device)
        t_emb = sinusoidal_time_embedding(t.view(b, 1), dim=self.time_embed_dim)

        if coords is None:
            coords = torch.zeros((b, txt.shape[1], 2), device=x.device)
        
        # Condition node features on current position x_t
        coord_feat = self.coord_proj(coords)  # [B, N, hidden]
        node_feat = txt + coord_feat  # Fuse text with position info

        edges = []
        for edges_list in skeleton:
            if len(edges_list) == 0:
                edges.append(torch.zeros((0, 2), device=x.device, dtype=torch.long))
            else:
                edges.append(torch.tensor(edges_list, device=x.device, dtype=torch.long))
        skeleton_tensor = torch.nn.utils.rnn.pad_sequence(edges, batch_first=True, padding_value=0)
        self.last_skeleton = skeleton_tensor

        # Pool text features for global conditioning (AdaLN, idea.md Step 4)
        text_cond = txt.mean(dim=1)  # [B, hidden]

        v_pred, metric = self.vector_head(
            node_feat=node_feat,  # Use position-conditioned features
            memory=memory,
            skeleton=skeleton_tensor,
            mask=mask_s,
            t=t,
            text_cond=text_cond,
        )

        heatmap = None
        if self.with_heatmap:
            heatmap = self.heatmap_head(feat)

        # Shape like PoseHead output for downstream usage
        # Output velocity field v_pred for flow matching
        output = v_pred.unsqueeze(0)  # [1, B, N, 2] velocity field
        return output, coords, heatmap

    def get_loss(self,
                 output,
                 initial_proposals,
                 similarity_map,
                 target,
                 target_heatmap,
                 target_weight,
                 target_sizes):
        """Compute losses for coordinate prediction."""
        losses = {}
        
        # Heatmap loss only - coordinate losses are handled in detector
        if self.with_heatmap and similarity_map is not None and target_heatmap is not None:
            heatmap_loss = self._compute_heatmap_loss(similarity_map, target_heatmap, target_weight)
            losses["heatmap_loss"] = heatmap_loss
        
        return losses
    
    def _compute_heatmap_loss(self, similarity_map, target_heatmap, target_weight):
        """Compute heatmap loss between predicted and target heatmaps."""
        # similarity_map: [bs, num_query, h, w] or [bs, 1, h, w]
        # target_heatmap: [bs, num_query, sh, sw]
        # target_weight: [bs, num_query, 1]
        
        h, w = similarity_map.shape[-2:]
        similarity_map = similarity_map.sigmoid()
        
        # Handle case where similarity_map has 1 channel
        if similarity_map.shape[1] == 1:
            similarity_map = similarity_map.expand(-1, target_heatmap.shape[1], -1, -1)
        
        # Resize target heatmap to match similarity_map size
        target_heatmap = F.interpolate(target_heatmap, size=(h, w), mode='bilinear', align_corners=False)
        
        # Normalize target heatmap
        target_max = target_heatmap.flatten(2).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        target_heatmap = target_heatmap / (target_max + 1e-10)
        
        # Compute normalizer
        normalizer = target_weight.squeeze(-1).sum(dim=-1).clamp(min=1)  # [bs]
        
        # MSE loss
        l2_loss = F.mse_loss(similarity_map, target_heatmap, reduction='none')  # [bs, nq, h, w]
        l2_loss = l2_loss * target_weight[:, :, :, None]  # [bs, nq, h, w]
        l2_loss = l2_loss.flatten(2).sum(-1) / (h * w)  # [bs, nq]
        l2_loss = l2_loss.sum(-1) / normalizer  # [bs]
        
        return l2_loss.mean()

    def get_accuracy(self,
                     output,
                     target,
                     target_weight,
                     target_sizes,
                     height=256):
        """Calculate PCK accuracy.

        This mirrors the interface used by detectors in this repo.

        Args:
            output (torch.Tensor[NxKx2]): predicted keypoints in normalized coords.
            target (torch.Tensor[NxKx2]): gt keypoints in absolute coords.
            target_weight (torch.Tensor[NxKx1]): keypoint visibility weights.
            target_sizes (torch.Tensor[Nx1x2]): image sizes.
        """
        accuracy = dict()

        if not torch.is_tensor(output):
            return accuracy

        if target_sizes is None:
            scale = float(height)
            output = output * scale
            target_sizes_np = np.array([[scale, scale]], dtype=np.float32).repeat(
                output.shape[0], axis=0)
        else:
            if target_sizes.dim() == 3:
                target_sizes = target_sizes.squeeze(1)
            output = output * target_sizes[:, None, :]
            target_sizes_np = target_sizes.detach().cpu().numpy()

        output_np = output.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        target_weight_np = target_weight.squeeze(-1).long().detach().cpu().numpy()

        _, avg_acc, _ = keypoint_pck_accuracy(
            output_np,
            target_np,
            target_weight_np.astype(np.bool8),
            thr=0.2,
            normalize=target_sizes_np)
        accuracy['acc_pose'] = float(avg_acc)
        return accuracy

    def decode(self, img_metas, output, img_size, **kwargs):
        """Decode predicted keypoints to original image space.

        Kept compatible with the decoder logic used by `PoseHead`.
        `output` is expected to be normalized coordinates in [0, 1].
        """
        batch_size = len(img_metas)
        W, H = img_size

        if torch.is_tensor(output):
            output = output.detach().cpu().numpy()

        output = output * np.array([W, H])[None, None, :]

        # NOTE: Preserve existing repo behavior (even though this condition is odd).
        if 'bbox_id' or 'query_bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['query_center']
            s[i, :] = img_metas[i]['query_scale']
            image_paths.append(img_metas[i]['query_image_file'])
            if 'query_bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['query_bbox_score']).reshape(-1)
            if 'bbox_id' in img_metas[i]:
                bbox_ids.append(img_metas[i]['bbox_id'])
            elif 'query_bbox_id' in img_metas[i]:
                bbox_ids.append(img_metas[i]['query_bbox_id'])

        preds = np.zeros(output.shape)
        for i in range(output.shape[0]):
            preds[i] = transform_preds(
                output[i],
                c[i],
                s[i],
                [W, H],
                use_udp=self.test_cfg.get('use_udp', False))

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = 1.0
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {
            'preds': all_preds,
            'boxes': all_boxes,
            'image_paths': image_paths,
            'bbox_ids': bbox_ids,
        }
        return result
