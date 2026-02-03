import numpy as np
import torch
import torch.nn.functional as F
from mmpose.models import builder
from mmpose.models.builder import POSENETS
from mmpose.models.detectors.base import BasePose
from transformers import BertTokenizerFast

from models.models.backbones.swin_utils import load_pretrained
import os
import clip
# import open_clip
# from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from transformers import BertTokenizer, BertModel
import random
from models.models.flow.riemannian_flow_matching import (
    FlowODEIntegrator,
    RFMConfig,
    laplacian_metric,
    riemannian_flow_loss,
)


def get_fully_connected_graph(visible_points):
    from itertools import combinations
    nodes = np.where((visible_points == np.array([1., 1., 0.])).all(axis=1))[0]
    edges = list(combinations(nodes, 2))
    edges = [[e1, e2] for e1, e2 in edges]
    return edges


def keyness_guidance(x: torch.Tensor, heatmap: torch.Tensor, lambda_: float = 0.1) -> torch.Tensor:
    """Compute Keyness potential gradient for ODE guidance (idea.md Step 3).
    
    During inference, we guide the flow towards keyness peaks:
    v_guided = v_pred - lambda * grad_x(-H(x))
    
    Uses numerical gradient to avoid autograd issues during inference.
    
    Args:
        x: Current coordinates [B, N, 2] in normalized [0,1] space
        heatmap: Keyness heatmap [B, 1, H, W] or [B, N, H, W]
        lambda_: Guidance strength
        
    Returns:
        Gradient adjustment [B, N, 2]
    """
    B, N, _ = x.shape
    _, C, H, W = heatmap.shape
    eps = 1.0 / H  # Numerical gradient step size
    
    def sample_heatmap(coords):
        """Sample heatmap at coordinates using grid_sample."""
        # Convert from [0,1] to [-1,1] for grid_sample
        grid = coords * 2 - 1  # [B, N, 2]
        grid = grid.unsqueeze(1)  # [B, 1, N, 2]
        sampled = F.grid_sample(heatmap, grid, mode='bilinear', 
                                padding_mode='border', align_corners=True)
        return sampled.squeeze(2).mean(dim=1)  # [B, N]
    
    # Compute numerical gradient using central difference
    # Gradient in x direction
    x_plus = x.clone()
    x_plus[..., 0] = (x_plus[..., 0] + eps).clamp(0, 1)
    x_minus = x.clone()
    x_minus[..., 0] = (x_minus[..., 0] - eps).clamp(0, 1)
    grad_x = (sample_heatmap(x_plus) - sample_heatmap(x_minus)) / (2 * eps)  # [B, N]
    
    # Gradient in y direction
    y_plus = x.clone()
    y_plus[..., 1] = (y_plus[..., 1] + eps).clamp(0, 1)
    y_minus = x.clone()
    y_minus[..., 1] = (y_minus[..., 1] - eps).clamp(0, 1)
    grad_y = (sample_heatmap(y_plus) - sample_heatmap(y_minus)) / (2 * eps)  # [B, N]
    
    # Stack to get full gradient
    grad = torch.stack([grad_x, grad_y], dim=-1)  # [B, N, 2]
    
    # Return gradient ascent direction (towards higher keyness)
    return lambda_ * grad


@POSENETS.register_module()
class FlowPoseModel(BasePose):
    """Few-shot keypoint detectors.
    Args:
        keypoint_head (dict): Keypoint head to process feature.
        encoder_config (dict): Config for encoder. Default: None.
        pretrained (str): Path to the pretrained image models.
        text_pretrained (str): Path to the pretrained text models.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
    """

    def __init__(self,
                 keypoint_head,
                 encoder_config,
                 pretrained=False,
                 text_pretrained=False,
                 finetune_text_pretrained=False,
                 rfm_cfg=None,
                 align_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.backbone, self.backbone_type = self.init_backbone(pretrained, encoder_config)
        self.finetune_text_pretrained = finetune_text_pretrained
        self.text_backbone, self.tokenizer, self.text_backbone_type = self.init_text_backbone(text_pretrained)
        self.keypoint_head = builder.build_head(keypoint_head)
        self.keypoint_head.init_weights()
        self.train_cfg = train_cfg if train_cfg is not None else {}
        self.test_cfg = test_cfg if test_cfg is not None else {}
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatMap')
        self.use_flow_ode = self.test_cfg.get('use_flow_ode', True)
        self.rfm_cfg = RFMConfig(**rfm_cfg) if rfm_cfg is not None else RFMConfig()
        self.flow_solver = FlowODEIntegrator(self.rfm_cfg) if self.use_flow_ode else None
        # align_cfg is accepted but not used in Rectified Flow (kept for config compatibility)
        self.align_cfg = align_cfg


    def init_text_backbone(self, text_pretrained):
        if "ViT" in text_pretrained:
            text_backbone, _ = clip.load(
                text_pretrained,
                download_root=os.getenv('TORCH_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'torch')),
                jit=False)
            tokenizer = clip.tokenize
            text_backbone_type = "clip"

            # Freeze all parameters of the visual backbone
            for param in text_backbone.visual.parameters():
                param.requires_grad = False

        elif "gte" in text_pretrained:
            is_local_path = os.path.isdir(text_pretrained)
            tokenizer = AutoTokenizer.from_pretrained(text_pretrained, local_files_only=is_local_path)
            text_backbone = AutoModel.from_pretrained(text_pretrained, trust_remote_code=True, local_files_only=is_local_path)
            text_backbone_type = "gte"

        elif "bert-base-multilingual" in text_pretrained:
            tokenizer = BertTokenizer.from_pretrained(text_pretrained)
            text_backbone = BertModel.from_pretrained(text_pretrained)
            text_backbone_type = "bert-multilingual"

        self.text_backbone_device = "cuda" if torch.cuda.is_available() else "cpu"
        text_backbone.to(device=self.text_backbone_device)

        if self.finetune_text_pretrained:
            if text_backbone_type == "clip":
                # https://github.com/openai/CLIP/issues/57
                for p in text_backbone.parameters():
                    p.data = p.data.float()
            text_backbone.train()
        else:
            text_backbone.eval()

        return text_backbone, tokenizer, text_backbone_type


    def init_backbone(self, pretrained, encoder_config):
        if 'swin' in pretrained:
            encoder_sample = builder.build_backbone(encoder_config)
            if '.pth' in pretrained:
                load_pretrained(pretrained, encoder_sample, logger=None)
            backbone = 'swin'
        elif 'dino' in pretrained:
            if 'dinov2' in pretrained:
                repo = 'facebookresearch/dinov2'
                backbone = 'dinov2'
            else:
                repo = 'facebookresearch/dino:main'
                backbone = 'dino'
            encoder_sample = torch.hub.load(repo, pretrained)
        elif 'resnet' in pretrained:
            pretrained = 'torchvision://resnet50'
            encoder_config = dict(type='ResNet', depth=50, out_indices=(3,))
            encoder_sample = builder.build_backbone(encoder_config)
            encoder_sample.init_weights(pretrained)
            backbone = 'resnet50'
        else:
            raise NotImplementedError(f'backbone {pretrained} not supported')
        return encoder_sample, backbone

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        self.backbone.init_weights(pretrained)
        self.encoder_query.init_weights(pretrained)
        self.keypoint_head.init_weights()

    def forward(self,
                img_s,
                img_q,
                target_s=None,
                target_weight_s=None,
                target_q=None,
                target_weight_q=None,
                img_metas=None,
                return_loss=True,
                **kwargs):
        """Defines the computation performed at every call."""

        if return_loss:
            return self.forward_train(img_s, target_s, target_weight_s, img_q,
                                      target_q, target_weight_q, img_metas,
                                      **kwargs)
        else:
            return self.forward_test(img_s, target_s, target_weight_s, img_q,
                                     target_q, target_weight_q, img_metas,
                                     **kwargs)

    def forward_dummy(self, img_s, target_s, target_weight_s, img_q, target_q,
                      target_weight_q, img_metas, **kwargs):
        return self.predict(
            img_s, target_s, target_weight_s, img_q, img_metas)

    def _get_centered_support_pose(self, img_metas, device, img_size, mask_s=None):
        """Get support pose (shot 0), normalized and centered to visible points' centroid.
        
        Args:
            img_metas: List of image meta info dicts.
            device: Torch device.
            img_size: Tensor [w, h] for normalization.
            mask_s: Optional visibility mask [B, N, 1].
            
        Returns:
            x0_centered: [B, N, 2] normalized coords centered at image center (0.5, 0.5).
        """
        # Get raw support keypoints and normalize to [0, 1]
        sample_keypoints = self.parse_keypoints_from_img_meta(
            img_metas, device, keyword='sample')  # [B, num_shots, N, 2]
        x0 = sample_keypoints[:, 0, :, :] / img_size  # [B, N, 2] in [0, 1]
        
        if mask_s is not None:
            # Center to visible points' centroid
            mask = mask_s.squeeze(-1) if mask_s.dim() == 3 else mask_s  # [B, N]
            sum_vis = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
            centroid = (x0 * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / sum_vis.unsqueeze(-1)  # [B, 1, 2]
        else:
            centroid = x0.mean(dim=1, keepdim=True)  # [B, 1, 2]
        
        # Translate so centroid is at image center (0.5, 0.5)
        x0_centered = x0 - centroid + 0.5
        
        return x0_centered.clamp(0.0, 1.0)

    def forward_train(self,
                      img_s,
                      target_s,
                      target_weight_s,
                      img_q,
                      target_q,
                      target_weight_q,
                      img_metas,
                      **kwargs):

        """Rectified Flow training: Support Pose -> Query Pose."""
        bs, _, h, w = img_q.shape

        # Parse target keypoints
        target_keypoints = self.parse_keypoints_from_img_meta(img_metas, img_q.device, keyword='query')
        
        # Extract features
        mask_s = target_weight_s[0]
        max_points = mask_s.shape[1]
        skeleton = [i['sample_skeleton'][0] for i in img_metas]
        all_shots_point_descriptions = self.extract_text_features(img_metas, max_points, mask_s)
        feature_q, feature_s = self.extract_image_features(img_s, img_q)
        
        img_size = torch.tensor([w, h], device=img_q.device).float()
        
        # === Rectified Flow: Support Pose -> Query Pose ===
        # x0 = Support Pose (centered) - SAME as test time!
        # x1 = Query Pose (GT)
        x0 = self._get_centered_support_pose(img_metas, img_q.device, img_size, mask_s)
        x1 = target_keypoints / img_size  # [B, N, 2] in [0,1]
        
        # Visibility mask: only train on points visible in BOTH support and query
        valid_mask = target_weight_q.squeeze(-1) * mask_s.squeeze(-1)  # [B, N]
        
        # Target velocity (constant for linear interpolation)
        v_target = x1 - x0  # [B, N, 2]
        
        # Sample time t - biased towards 0 for better velocity learning at low t
        # Using sqrt gives more weight to low t values where velocity quality matters
        t = torch.rand((bs, 1, 1), device=img_q.device) ** 0.5  # Biased towards 0
        
        # Interpolate to get x_t
        x_t = (1 - t) * x0 + t * x1  # [B, N, 2]
        
        # Predict velocity field v(x_t, t)
        output, _, similarity_map = self.keypoint_head(
            feature_q, feature_s, target_s, mask_s,
            skeleton, all_shots_point_descriptions,
            coords=x_t, t=t)
        
        v_pred = output[-1]  # [B, N, 2] predicted velocity
        
        # Predicted coordinates (for accuracy calculation)
        pred_coords = x_t + v_pred * (1 - t)
        
        target_sizes = torch.tensor([w, h], device=img_q.device).float().unsqueeze(0).repeat(bs, 1, 1)

        # === Compute Losses ===
        losses = dict()
        if self.with_keypoint:
            mask = valid_mask
            normalizer = mask.sum(dim=-1).clamp(min=1)  # [B]
            
            # === 1. Riemannian Flow Matching Loss ===
            skeleton_edges = [meta['sample_skeleton'][0] for meta in img_metas]
            skeleton_tensors = [
                torch.tensor(edges, device=img_q.device, dtype=torch.long) if len(edges) > 0
                else torch.zeros((0, 2), device=img_q.device, dtype=torch.long)
                for edges in skeleton_edges
            ]
            skeleton_tensor = torch.nn.utils.rnn.pad_sequence(skeleton_tensors, batch_first=True, padding_value=0)
            
            # Compute Laplacian metric M(G)
            M = laplacian_metric(skeleton_tensor, mask, num_points=max_points)  # [B, N, N]
            
            # Velocity difference
            diff = v_pred - v_target  # [B, N, 2]
            
            # Riemannian weighted loss
            M_diff = torch.bmm(M, diff)  # [B, N, 2]
            rfm_loss = (diff * M_diff * mask.unsqueeze(-1)).sum(dim=[1, 2]) / normalizer
            losses['rfm_loss'] = rfm_loss.mean()
            
            # Simple flow loss for stability
            flow_loss = F.mse_loss(v_pred, v_target, reduction='none')  # [B, N, 2]
            flow_loss = (flow_loss.sum(dim=-1) * mask).sum(dim=-1) / normalizer
            losses['flow_loss'] = flow_loss.mean()
            
            # 2. Auxiliary L1 loss on predicted coordinates
            kpt_loss = F.l1_loss(pred_coords, x1, reduction='none')
            kpt_loss = (kpt_loss.sum(dim=-1) * mask).sum(dim=-1) / normalizer
            losses['kpt_loss'] = kpt_loss.mean() * 0.5
            
            # 3. Heatmap loss (if enabled)
            if similarity_map is not None and target_q is not None:
                head_losses = self.keypoint_head.get_loss(
                    output, x_t, similarity_map, target_keypoints,
                    target_q, target_weight_q, target_sizes)
                if 'heatmap_loss' in head_losses:
                    losses['heatmap_loss'] = head_losses['heatmap_loss']
            
            # === 4. Direct t=0 Velocity Loss (NEW: force correct velocity at start) ===
            t_zero = torch.zeros((bs, 1, 1), device=img_q.device)
            out_zero, _, _ = self.keypoint_head(
                feature_q, feature_s, target_s, mask_s,
                skeleton, all_shots_point_descriptions,
                coords=x0, t=t_zero)
            v_pred_zero = out_zero[-1]
            v_zero_loss = F.mse_loss(v_pred_zero, v_target, reduction='none')
            v_zero_loss = (v_zero_loss.sum(dim=-1) * mask).sum(dim=-1) / normalizer
            losses['v_zero_loss'] = v_zero_loss.mean() * 2.0  # Higher weight for t=0
            
            # === 5. ODE Loss (for train/test consistency) ===
            if self.train_cfg.get('with_ode_loss', True):
                x_ode = x0.clone()  # Start from SAME x0 as flow matching
                num_ode_steps = self.train_cfg.get('num_ode_steps', 10)  # Match test (was 3)
                dt = 1.0 / num_ode_steps
                for step in range(num_ode_steps):
                    t_ode = torch.full((bs, 1, 1), step * dt, device=img_q.device)
                    out_ode, _, _ = self.keypoint_head(
                        feature_q, feature_s, target_s, mask_s,
                        skeleton, all_shots_point_descriptions,
                        coords=x_ode, t=t_ode)
                    v_ode = out_ode[-1]
                    x_ode = x_ode + v_ode * dt
                
                ode_loss = F.l1_loss(x_ode, x1, reduction='none')
                ode_loss = (ode_loss.sum(dim=-1) * mask).sum(dim=-1) / normalizer
                losses['ode_loss'] = ode_loss.mean()
            
            # === 6. Refinement Layer Loss (CAPEx-style multi-layer supervision) ===
            if hasattr(self.keypoint_head, 'num_refine_layers') and self.keypoint_head.num_refine_layers > 0:
                # Get features for refinement
                feat = self.keypoint_head.img_proj(feature_q)
                masks_pos = feat.new_zeros((bs, feat.shape[2], feat.shape[3]), dtype=torch.bool)
                pos_embed = self.keypoint_head.positional_encoding(masks_pos)
                memory = (feat + pos_embed).flatten(2).transpose(1, 2)
                
                # Joint training: allow gradient flow to flow matching
                x_refine = pred_coords
                
                # Loss weight for refinement (lower to avoid dominating training)
                refine_weight = self.train_cfg.get('refine_loss_weight', 0.1)
                
                # Apply refinement and compute loss for each layer
                for layer_idx, layer in enumerate(self.keypoint_head.refine_layers):
                    delta = layer(x_refine, memory, self.keypoint_head.coord_proj, mask_s)
                    x_refine = x_refine + delta
                    
                    # L1 loss for this layer (with reduced weight)
                    layer_loss = F.l1_loss(x_refine, x1, reduction='none')
                    layer_loss = (layer_loss.sum(dim=-1) * mask).sum(dim=-1) / normalizer
                    losses[f'refine_loss_layer{layer_idx}'] = layer_loss.mean() * refine_weight
            
            # Accuracy using single-step prediction
            keypoint_accuracy = self.keypoint_head.get_accuracy(
                pred_coords.detach().clamp(0.0, 1.0),
                target_keypoints,
                target_weight_q,
                target_sizes,
                height=h)
            losses.update(keypoint_accuracy)

        return losses


    def forward_test(self,
                     img_s,
                     target_s,
                     target_weight_s,
                     img_q,
                     target_q,
                     target_weight_q,
                     img_metas=None,
                     **kwargs):

        """Rectified Flow inference: Integrate ODE from Support Pose."""
        batch_size, _, img_height, img_width = img_q.shape

        # Extract features
        mask_s = target_weight_s[0]
        max_points = mask_s.shape[1]
        skeleton = [i['sample_skeleton'][0] for i in img_metas]
        all_shots_point_descriptions = self.extract_text_features(
            img_metas, max_points, mask_s)
        feature_q, feature_s = self.extract_image_features(img_s, img_q)
        
        img_size = torch.tensor([img_width, img_height], device=img_q.device).float()
        
        # === Rectified Flow: x0 = Support Pose (centered) - SAME as training! ===
        x0 = self._get_centered_support_pose(img_metas, img_q.device, img_size, mask_s)
        x = x0.clone()
        
        # ODE integration from t=0 to t=1
        num_steps = self.rfm_cfg.num_timesteps if hasattr(self, 'rfm_cfg') else 10
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t = torch.full((batch_size, 1, 1), step * dt, device=img_q.device)
            
            # Predict velocity field v(x, t)
            output, _, heatmap = self.keypoint_head(
                feature_q, feature_s, target_s, mask_s,
                skeleton, all_shots_point_descriptions,
                coords=x, t=t)
            
            v_pred = output[-1]  # [B, N, 2] velocity
            
            # Optional: Keyness guidance
            if heatmap is not None and self.test_cfg.get('use_keyness_guidance', False):
                lambda_guidance = self.test_cfg.get('keyness_lambda', 0.1)
                guidance = keyness_guidance(x, heatmap, lambda_=lambda_guidance)
                v_pred = v_pred + guidance
            
            # Euler step
            x = x + v_pred * dt
        
        # === Iterative Refinement (CAPEx-style) ===
        if hasattr(self.keypoint_head, 'num_refine_layers') and self.keypoint_head.num_refine_layers > 0:
            # Get memory for refinement
            feat = self.keypoint_head.img_proj(feature_q)
            masks_pos = feat.new_zeros((batch_size, feat.shape[2], feat.shape[3]), dtype=torch.bool)
            pos_embed = self.keypoint_head.positional_encoding(masks_pos)
            memory = (feat + pos_embed).flatten(2).transpose(1, 2)
            
            # Apply refinement layers
            x, _ = self.keypoint_head.refine(x, memory, mask_s)
        
        # Clamp final result to valid range
        x = x.clamp(0.0, 1.0)
        
        predicted_pose = x.detach().cpu().numpy()

        result = {}
        if self.with_keypoint:
            keypoint_result = self.keypoint_head.decode(
                img_metas, predicted_pose, img_size=[img_width, img_height])
            result.update(keypoint_result)

        result.update({"points": predicted_pose})
        result.update({"sample_image_file": img_metas[0]['sample_image_file']})

        return result

    def predict(self,
                img_s,
                target_s,
                target_weight_s,
                img_q,
                img_metas=None,
                training=False,
                target_keypoints=None):
        """Predict with flow matching.
        
        For Conditional Flow Matching:
        - Training: sample t ~ U[0,1], compute x_t = (1-t)*x0 + t*x1
          The target velocity is v = x1 - x0 (constant across t)
          Network learns to predict v(x_t, t) = x1 - x0
        - Inference: integrate ODE from t=0 to t=1 using learned velocity field
        
        Args:
            training: If True, sample t and interpolate x_t for training.
            target_keypoints: [B, N, 2] target keypoints in absolute coords (for training).
        
        Returns:
            output: predicted velocity field [1, B, N, 2]
            x0: noise starting point [B, N, 2]
            similarity_map: heatmap from head
            mask_s: visibility mask
            x1: normalized target keypoints [B, N, 2] (None for inference)
            t: sampled timestep [B, 1, 1]
        """
        batch_size, _, img_height, img_width = img_q.shape
        mask_s = target_weight_s[0]
        max_points = mask_s.shape[1]
        assert [i['sample_skeleton'][0] != i['query_skeleton'] for i in img_metas]
        skeleton = [i['sample_skeleton'][0] for i in img_metas]

        all_shots_point_descriptions = self.extract_text_features(img_metas, max_points, mask_s)
        feature_q, feature_s = self.extract_image_features(img_s, img_q)

        # Flow matching: x0 = noise, x1 = target
        img_size = torch.tensor([img_width, img_height], device=img_q.device).float()
        
        # Sample noise as x0 - use standard normal for better coverage
        x0 = torch.randn((batch_size, max_points, 2), device=img_q.device) * 0.3 + 0.5
        x0 = x0.clamp(0.05, 0.95)  # Keep in valid range with margin
        
        if training and target_keypoints is not None:
            # Training: sample t uniformly, including near t=0 for inference consistency
            t = torch.rand((batch_size, 1, 1), device=img_q.device)
            # Allow t to be very close to 0 so model learns to predict from noise
            t = t.clamp(min=1e-9, max=1.0)
            
            # Normalize target to [0, 1]
            x1 = target_keypoints / img_size  # [B, N, 2]
            
            # Interpolate: x_t = (1-t) * x0 + t * x1
            x_t = (1 - t) * x0 + t * x1
            coords = x_t
        else:
            # Inference: start from t=0 (handled in forward_test with ODE integration)
            t = torch.zeros((batch_size, 1, 1), device=img_q.device)
            x1 = None
            coords = x0

        output, initial_proposals, similarity_map = self.keypoint_head(
            feature_q, feature_s, target_s, mask_s,
            skeleton, all_shots_point_descriptions,
            coords=coords, t=t)
        
        return output, x0, similarity_map, mask_s, x1, t

    def extract_image_features(self, img_s, img_q):
        if self.backbone_type == 'swin':
            feature_q = self.backbone.forward_features(img_q)  # [bs, C, h, w]
            # feature_s = [self.backbone.forward_features(img) for img in img_s]
            feature_s = None
        elif self.backbone_type == 'dino':
            batch_size, _, img_height, img_width = img_q.shape
            feature_q = self.backbone.get_intermediate_layers(img_q, n=1)[0][:, 1:] \
                .reshape(batch_size, img_height // 8, img_width // 8, -1).permute(0, 3, 1, 2)  # [bs, 3, h, w]
            feature_s = [self.backbone.get_intermediate_layers(img, n=1)[0][:, 1:].
                         reshape(batch_size, img_height // 8, img_width // 8, -1).permute(0, 3, 1, 2) for img in img_s]
        elif self.backbone_type == 'dinov2':
            batch_size, _, img_height, img_width = img_q.shape
            feature_q = self.backbone.get_intermediate_layers(img_q, n=1, reshape=True)[0]  # [bs, c, h, w]
            feature_s = [self.backbone.get_intermediate_layers(img, n=1, reshape=True)[0] for img in img_s]
        elif self.backbone_type == 'clip':
            with torch.no_grad():
                feature_q = self.backbone.model(img_q)
                feature_s = None
        else:
            feature_s = [self.backbone(img) for img in img_s]
            feature_q = self.encoder_query(img_q)

        return feature_q, feature_s


    def extract_text_features(self, img_metas, max_points, mask_s):
        with torch.set_grad_enabled(self.finetune_text_pretrained):
            all_shots_point_descriptions = []
            for shot in range(len(img_metas[0]['sample_point_descriptions'])):
                support_descriptions = [i['sample_point_descriptions'][shot] for i in img_metas]

                # ignore non-visible points
                # support_descriptions = [description[mask[:len(description)].view(-1) == 1] for mask, description in zip(mask_s.cpu(), support_descriptions)]
                support_descriptions = [description[list(mask[:len(description)].view(-1) == 1)] for mask, description in
                                        zip(mask_s.cpu(), support_descriptions)]

                all_points = [point for description in support_descriptions for point in description]
                # # CLIP
                if self.text_backbone_type == "clip":
                    tokens = self.tokenizer(all_points).to(device=self.text_backbone_device)
                    all_descriptions = self.text_backbone.encode_text(tokens)
                    all_descriptions = all_descriptions / all_descriptions.norm(dim=1, keepdim=True).to(dtype=torch.float32)

                # TRANSFORMERS
                elif self.text_backbone_type == "gte" or self.text_backbone_type == "bert-multilingual":
                    tokens = self.tokenizer(all_points, max_length=77, padding=True, truncation=True, return_tensors='pt').to(device=self.text_backbone_device)
                    all_descriptions = self.text_backbone(**tokens)
                    all_descriptions = all_descriptions.last_hidden_state[:, 0]
                    all_descriptions = torch.nn.functional.normalize(all_descriptions, p=2, dim=1)

                # Divide it back into a list of lists with original lengths
                batch_padded_tensors = []
                start_index = 0
                for i, description in enumerate(support_descriptions):
                    end_index = start_index + len(description)
                    # # pad all unused points with 0, up to max_points (default is 100)
                    padded_tensor = torch.zeros(max_points, all_descriptions.shape[-1]).to(device=self.text_backbone_device).detach()
                    padded_tensor[mask_s[i].view(-1) == 1] = all_descriptions[start_index:end_index]
                    # padded_tensor[:len(description)] = all_descriptions[start_index:end_index]

                    batch_padded_tensors.append(padded_tensor)
                    start_index = end_index
                all_shots_point_descriptions.append(torch.stack(batch_padded_tensors, dim=0))

            return torch.mean(torch.stack(all_shots_point_descriptions, dim=0), 0)

    def parse_keypoints_from_img_meta(self, img_meta, device, keyword='query'):
        """Parse keypoints from the img_meta.

        Args:
            img_meta (dict): Image meta info.
            device (torch.device): Device of the output keypoints.
            keyword (str): 'query' or 'sample'. Default: 'query'.

        Returns:
            Tensor: Keypoints coordinates of query images.
        """

        if keyword == 'query':
            query_kpt = torch.stack([
                torch.tensor(info[f'{keyword}_joints_3d']).to(device)
                for info in img_meta
            ], dim=0)[:, :, :2]  # [bs, num_query, 2]
        else:
            query_kpt = []
            for info in img_meta:
                if isinstance(info[f'{keyword}_joints_3d'][0], torch.Tensor):
                    samples = torch.stack(info[f'{keyword}_joints_3d'])
                else:
                    samples = np.array(info[f'{keyword}_joints_3d'])
                query_kpt.append(torch.tensor(samples).to(device)[:, :, :2])
            query_kpt = torch.stack(query_kpt, dim=0)  # [bs, , num_samples, num_query, 2]
        return query_kpt

    def compute_rfm_loss(self,
                         output,
                         initial_proposals,
                         target_keypoints,
                         target_weight_q,
                         target_sizes,
                         img_metas):
        """Compute Riemannian Flow Matching loss."""
        batch_size, num_points = initial_proposals.shape[:2]
        x0 = initial_proposals
        target_sizes = target_sizes.to(output.device)  # [bs, 1, 2]
        x1 = target_keypoints / target_sizes  # normalize to [0,1]

        vis_mask = (target_weight_q.squeeze(-1) > 0).float()
        skeleton_edges = [meta['sample_skeleton'][0] for meta in img_metas]
        # Pad skeleton edges to handle variable lengths across batch
        skeleton_tensors = [
            torch.tensor(edges, device=output.device, dtype=torch.long) if len(edges) > 0 
            else torch.zeros((0, 2), device=output.device, dtype=torch.long)
            for edges in skeleton_edges
        ]
        skeleton = torch.nn.utils.rnn.pad_sequence(skeleton_tensors, batch_first=True, padding_value=0)
        metric = laplacian_metric(skeleton, vis_mask, num_points=num_points)

        t = torch.rand((batch_size, 1, 1), device=output.device)
        t = t * (1.0 - self.rfm_cfg.t_epsilon) + self.rfm_cfg.t_epsilon

        v_pred = output[-1] - x0
        loss, _ = riemannian_flow_loss(v_pred, x0, x1, metric, t)
        return {"detector_rfm_loss": loss*1e-5}

    # optional: inference refinement helper (not called unless you wire it)
    def flow_refine(self, x0, vector_field_fn, guidance_fn=None):
        if self.flow_solver is None:
            return x0
        return self.flow_solver.integrate(x0, vector_field_fn, guidance=guidance_fn)

    # UNMODIFIED
    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_limb_color=None,
                    radius=4,
                    text_color=(255, 0, 0),
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_limb_color (np.array[Mx3]): Color of M limbs.
                If None, do not draw limbs.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """

        img = mmcv.imread(img)
        img = img.copy()
        img_h, img_w, _ = img.shape

        bbox_result = []
        pose_result = []
        for res in result:
            bbox_result.append(res['bbox'])
            pose_result.append(res['keypoints'])

        if len(bbox_result) > 0:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            mmcv.imshow_bboxes(
                img,
                bboxes,
                colors=bbox_color,
                top_k=-1,
                thickness=thickness,
                show=False,
                win_name=win_name,
                wait_time=wait_time,
                out_file=None)

            for person_id, kpts in enumerate(pose_result):
                # draw each point on image
                if pose_kpt_color is not None:
                    assert len(pose_kpt_color) == len(kpts), (
                        len(pose_kpt_color), len(kpts))
                    for kid, kpt in enumerate(kpts):
                        x_coord, y_coord, kpt_score = int(kpt[0]), int(
                            kpt[1]), kpt[2]
                        if kpt_score > kpt_score_thr:
                            img_copy = img.copy()
                            r, g, b = pose_kpt_color[kid]
                            cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                       radius, (int(r), int(g), int(b)), -1)
                            transparency = max(0, min(1, kpt_score))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)

                # draw limbs
                if skeleton is not None and pose_limb_color is not None:
                    assert len(pose_limb_color) == len(skeleton)
                    for sk_id, sk in enumerate(skeleton):
                        pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1,
                        1]))
                        pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1,
                        1]))
                        if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                                and pos1[1] < img_h and pos2[0] > 0
                                and pos2[0] < img_w and pos2[1] > 0
                                and pos2[1] < img_h
                                and kpts[sk[0] - 1, 2] > kpt_score_thr
                                and kpts[sk[1] - 1, 2] > kpt_score_thr):
                            img_copy = img.copy()
                            X = (pos1[0], pos2[0])
                            Y = (pos1[1], pos2[1])
                            mX = np.mean(X)
                            mY = np.mean(Y)
                            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                            angle = math.degrees(
                                math.atan2(Y[0] - Y[1], X[0] - X[1]))
                            stickwidth = 2
                            polygon = cv2.ellipse2Poly(
                                (int(mX), int(mY)),
                                (int(length / 2), int(stickwidth)), int(angle),
                                0, 360, 1)

                            r, g, b = pose_limb_color[sk_id]
                            cv2.fillConvexPoly(img_copy, polygon,
                                               (int(r), int(g), int(b)))
                            transparency = max(
                                0,
                                min(
                                    1, 0.5 *
                                       (kpts[sk[0] - 1, 2] + kpts[sk[1] - 1, 2])))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)

        show, wait_time = 1, 1
        if show:
            height, width = img.shape[:2]
            max_ = max(height, width)

            factor = min(1, 800 / max_)
            enlarge = cv2.resize(
                img, (0, 0),
                fx=factor,
                fy=factor,
                interpolation=cv2.INTER_CUBIC)
            imshow(enlarge, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img
