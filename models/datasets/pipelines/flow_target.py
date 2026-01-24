"""
Flow-matching target generation for keypoints.

Instead of Gaussian heatmaps, this pipeline stage emits:
- x0 (support coords), x1 (query coords) normalized to [0,1]
- visibility mask

Reference pattern for flow-matching target prep:
https://github.com/facebookresearch/flow-matching/blob/main/fm/data/transforms.py
"""

import numpy as np
import torch
from mmpose.datasets.builder import PIPELINES


@PIPELINES.register_module()
class GenerateFlowTargets:
    """Generate flow targets (x0, x1, mask) for flow-matching training."""

    def __init__(self, use_skeleton_weights=False, skeleton_weight_strength=1.0):
        self.use_skeleton_weights = use_skeleton_weights
        self.skeleton_weight_strength = skeleton_weight_strength

    def _compute_align_weight(self, skeleton, num_joints, vis_mask):
        weight = np.ones((num_joints,), dtype=np.float32)
        if self.use_skeleton_weights and skeleton is not None and len(skeleton) > 0:
            edges = np.array(skeleton, dtype=np.int64)
            if edges.ndim == 2 and edges.shape[1] == 2 and edges.size > 0:
                max_idx = edges.max()
                if max_idx == num_joints:
                    edges = edges - 1
                deg = np.zeros((num_joints,), dtype=np.float32)
                for u, v in edges:
                    if 0 <= u < num_joints:
                        deg[u] += 1.0
                    if 0 <= v < num_joints:
                        deg[v] += 1.0
                weight = weight + self.skeleton_weight_strength * deg
        weight = weight * vis_mask
        return weight

    def __call__(self, results):
        joints_3d = results['joints_3d']  # query joints
        joints_3d_visible = results['joints_3d_visible']
        ann_info = results['ann_info']
        image_size = ann_info['image_size']
        heatmap_size = ann_info['heatmap_size']
        w, h = float(image_size[0]), float(image_size[1])
        W, H = int(heatmap_size[0]), int(heatmap_size[1])
        num_joints = len(joints_3d)

        # Normalize query coords
        x1 = joints_3d[:, :2] / np.array([w, h], dtype=np.float32)
        mask1 = (joints_3d_visible[:, 0] > 0).astype(np.float32)

        # For support, reuse the same joints if provided, else zeros
        if 'support_joints_3d' in results:
            x0_raw = results['support_joints_3d'][:, :2]
            vis0 = (results['support_joints_3d_visible'][:, 0] > 0).astype(np.float32)
            x0 = x0_raw / np.array([w, h], dtype=np.float32)
            mask0 = vis0
        else:
            x0 = np.zeros_like(x1)
            mask0 = np.zeros_like(mask1)

        mask = (mask0 * mask1).astype(np.float32)

        results['flow_x0'] = torch.tensor(x0, dtype=torch.float32)
        results['flow_x1'] = torch.tensor(x1, dtype=torch.float32)
        results['flow_mask'] = torch.tensor(mask, dtype=torch.float32)[:, None]  # [N,1]

        skeleton = results.get('skeleton', None)
        align_weight = self._compute_align_weight(skeleton, num_joints, mask1)
        results['align_weight'] = align_weight[:, None]

        # Create placeholder target for pipeline compatibility.
        # Keep a correct visibility-based target_weight because downstream code
        # uses it as the point mask (support/query visibility, and later combined).
        results['target'] = np.zeros((num_joints, H, W), dtype=np.float32)
        results['target_weight'] = (joints_3d_visible[:, 0] > 0).astype(np.float32)[:, None]

        return results
