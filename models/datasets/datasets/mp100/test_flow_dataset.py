"""
Flow-friendly test dataset for MP-100.

Extends TestPoseDataset to emit flow tensors (x0, x1, mask, skeleton)
for evaluation/inference with flow matching models.

Reference pattern for flow datasets:
https://github.com/facebookresearch/flow-matching/blob/main/fm/data/datasets.py
"""

import torch
import os
import random
from collections import OrderedDict

import numpy as np
from mmcv.parallel import DataContainer as DC
from mmpose.datasets import DATASETS
from xtcocotools.coco import COCO
from .test_dataset import TestPoseDataset

@DATASETS.register_module()
class TestFlowPoseDataset(TestPoseDataset):
    """Adds flow fields to TestPoseDataset outputs."""

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        img_metas = data['img_metas'].data

        # Few-shot batches are merged into Xall with sample_*/query_* meta keys.
        # Fall back to legacy keys when running on non-merged datasets.
        if 'sample_joints_3d' in img_metas and 'query_joints_3d' in img_metas:
            sample_joints = img_metas['sample_joints_3d'][0]  # [N, 3]
            sample_vis = img_metas.get('sample_joints_3d_visible', [None])[0]
            query_joints = img_metas['query_joints_3d']  # [N, 3]
            query_vis = img_metas.get('query_joints_3d_visible', None)
        else:
            sample_joints = img_metas['joints_3d'][0]  # [N, 3]
            sample_vis = img_metas['joints_3d_visible'][0]  # [N, 3]
            query_joints = img_metas['joints_3d']  # [N, 3]
            query_vis = img_metas['joints_3d_visible']  # [N, 3]

        img_size = img_metas.get('query_image_size', None)
        if img_size is None:
            img_size = img_metas.get('image_size', None)
        if img_size is None:
            ann_info = img_metas.get('query_ann_info', None)
            if ann_info is None:
                ann_info = img_metas.get('ann_info', None)
            if ann_info is not None:
                img_size = ann_info.get('image_size', None)
        if img_size is None:
            img_size = np.array([256, 256], dtype=np.float32)
        w, h = float(img_size[0]), float(img_size[1])

        x0 = torch.tensor(sample_joints[:, :2] / np.array([w, h]), dtype=torch.float32)
        x1 = torch.tensor(query_joints[:, :2] / np.array([w, h]), dtype=torch.float32)
        if sample_vis is None:
            sample_vis = np.ones_like(sample_joints)
        if query_vis is None:
            query_vis = np.ones_like(query_joints)
        mask = torch.tensor((sample_vis[:, 0] > 0) & (query_vis[:, 0] > 0), dtype=torch.float32).unsqueeze(-1)

        skeleton_edges = img_metas.get('sample_skeleton', None)
        if skeleton_edges is None:
            skeleton_edges = img_metas.get('skeleton', None)
        if skeleton_edges is None:
            skeleton_edges = data.get('skeleton', [])
        if isinstance(skeleton_edges, list) and len(skeleton_edges) > 0 and isinstance(skeleton_edges[0], (list, tuple)) and len(skeleton_edges[0]) > 0 and isinstance(skeleton_edges[0][0], (list, tuple)):
            skeleton_edges = skeleton_edges[0]
        skeleton_edges = torch.tensor(skeleton_edges, dtype=torch.long) if len(skeleton_edges) > 0 else torch.zeros((0, 2), dtype=torch.long)

        data['flow'] = dict(
            x0=x0,
            x1=x1,
            mask=mask,
        )
        data['flow_skeleton'] = DC(skeleton_edges, cpu_only=True, stack=False)
        return data
