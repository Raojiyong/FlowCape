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

        # Create placeholder target for pipeline compatibility.
        # Keep a correct visibility-based target_weight because downstream code
        # uses it as the point mask (support/query visibility, and later combined).
        results['target'] = np.zeros((num_joints, H, W), dtype=np.float32)
        results['target_weight'] = (joints_3d_visible[:, 0] > 0).astype(np.float32)[:, None]

        return results

