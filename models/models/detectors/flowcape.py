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
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.backbone, self.backbone_type = self.init_backbone(pretrained, encoder_config)
        self.finetune_text_pretrained = finetune_text_pretrained
        self.text_backbone, self.tokenizer, self.text_backbone_type = self.init_text_backbone(text_pretrained)
        self.keypoint_head = builder.build_head(keypoint_head)
        self.keypoint_head.init_weights()
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = test_cfg
        self.target_type = test_cfg.get('target_type', 'GaussianHeatMap')  # GaussianHeatMap
        self.use_flow_ode = self.test_cfg.get('use_flow_ode', True)
        self.rfm_cfg = RFMConfig(**rfm_cfg) if rfm_cfg is not None else RFMConfig()
        self.flow_solver = FlowODEIntegrator(self.rfm_cfg) if self.use_flow_ode else None


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
        # self.encoder_query.init_weights(pretrained) # Removed as not defined
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
        """Get support pose (shot 0), normalized and centered to image center.
        
        Args:
            img_metas: List of image meta dicts.
            device: Tensor device.
            img_size: Tensor [W, H].
            mask_s: Visibility mask [B, N, 1] (Optional).
        """
        # Get raw support joints [B, num_samples, N, 2] -> take 1st shot [B, N, 2]
        support_kpts = self.parse_keypoints_from_img_meta(img_metas, device, keyword='sample')[:, 0]
        
        # Normalize to [0,1]
        x0 = support_kpts / img_size
        
        # Center alignment: Shift support pose so its center matches query image center (0.5, 0.5)
        # We calculate the center of the support pose using visible points if mask is provided
        if mask_s is not None:
            # mask_s: [B, N, 1]
            sum_vis = mask_s.sum(dim=1).clamp(min=1) # [B, 1]
            means = (x0 * mask_s).sum(dim=1) / sum_vis # [B, 2]
            means = means.unsqueeze(1) # [B, 1, 2]
        else:
            means = x0.mean(dim=1, keepdim=True) # [B, 1, 2]

        x0 = x0 - means + 0.5
        
        return x0.clamp(0.0, 1.0)



    def forward_train(self,
                      img_s,
                      target_s,
                      target_weight_s,
                      img_q,
                      target_q,
                      target_weight_q,
                      img_metas,
                      **kwargs):

        """Defines the computation performed at every call when training."""
        bs, _, h, w = img_q.shape

        # Parse target keypoints (Query)
        target_keypoints = self.parse_keypoints_from_img_meta(img_metas, img_q.device, keyword='query')
        
        # Extract features
        mask_s = target_weight_s[0]
        max_points = mask_s.shape[1]
        skeleton = [i['sample_skeleton'][0] for i in img_metas]
        all_shots_point_descriptions = self.extract_text_features(img_metas, max_points, mask_s)
        feature_q, feature_s = self.extract_image_features(img_s, img_q)
        
        img_size = torch.tensor([w, h], device=img_q.device).float()
        
        # === Corrected Riemannian Flow Matching (idea.md) ===
        # x0 = Support Pose (Centered) - Source distribution
        # x1 = Query Pose - Target distribution
        
        x0 = self._get_centered_support_pose(img_metas, img_q.device, img_size, mask_s)
        x1 = target_keypoints / img_size  # [B, N, 2] in [0,1]

        # Combine masks: We only train flow on points valid in both Support AND Query
        # If support points are invalid, x0 is garbage/zero.
        valid_mask = target_weight_q.squeeze(-1) * mask_s.squeeze(-1) # [B, N]
        
        # Target velocity (constant for linear interpolation)
        v_target = x1 - x0  # [B, N, 2]
        
        target_sizes = torch.tensor([w, h], device=img_q.device).float().unsqueeze(0).repeat(bs, 1, 1)

        # === Compute Losses ===
        losses = dict()
        if self.with_keypoint:
            # Use intersection mask for flow losses to ensure validity of source and target
            mask = valid_mask
            normalizer = mask.sum(dim=-1).clamp(min=1)  # [B]
            
            # === 1. Riemannian Flow Matching Loss (idea.md §2.2) ===

            # L_RFM = ||v_pred - v_target||^2_M(G) where M is Laplacian metric
            skeleton_edges = [meta['sample_skeleton'][0] for meta in img_metas]
            skeleton_tensors = [
                torch.tensor(edges, device=img_q.device, dtype=torch.long) if len(edges) > 0
                else torch.zeros((0, 2), device=img_q.device, dtype=torch.long)
                for edges in skeleton_edges
            ]
            skeleton_tensor = torch.nn.utils.rnn.pad_sequence(skeleton_tensors, batch_first=True, padding_value=0)
            
            # Compute Laplacian metric M(G)
            M = laplacian_metric(skeleton_tensor, mask, num_points=max_points)  # [B, N, N]
            
            # Sample multiple t for time-axis supervision
            num_t_samples = int(self.train_cfg.get('num_sampled_t', 1))
            num_t_samples = max(1, num_t_samples)

            rfm_loss_acc = 0.0
            flow_loss_acc = 0.0
            kpt_loss_acc = 0.0
            first_output = None
            first_x_t = None
            first_similarity_map = None

            for t_idx in range(num_t_samples):
                # Sample time t ~ U[0, 1]
                t = torch.rand((bs, 1, 1), device=img_q.device)

                # Interpolate to get x_t (Linear Geodesic approx)
                x_t = (1 - t) * x0 + t * x1  # [B, N, 2]

                # Predict velocity field v(x_t, t)
                # Note: We pass coords=x_t to condition the field on current geometry
                output, _, similarity_map = self.keypoint_head(
                    feature_q, feature_s, target_s, mask_s,
                    skeleton, all_shots_point_descriptions,
                    coords=x_t, t=t)

                v_pred = output[-1]  # [B, N, 2] predicted velocity

                # Predicted coordinates (Euler step from x_t to x1 using v_pred)
                pred_coords = x_t + v_pred * (1 - t)

                # Velocity difference
                diff = v_pred - v_target  # [B, N, 2]

                # Riemannian weighted loss: (v_pred - v_target)^T * M * (v_pred - v_target)
                # For each coordinate dimension
                M_diff = torch.bmm(M, diff)  # [B, N, 2]
                rfm_loss = (diff * M_diff * mask.unsqueeze(-1)).sum(dim=[1, 2]) / normalizer
                rfm_loss_acc = rfm_loss_acc + rfm_loss.mean()

                # Also keep simple flow loss for stability
                flow_loss = F.mse_loss(v_pred, v_target, reduction='none')  # [B, N, 2]
                flow_loss = (flow_loss.sum(dim=-1) * mask).sum(dim=-1) / normalizer
                flow_loss_acc = flow_loss_acc + flow_loss.mean()

                # 2. Auxiliary L1 loss on predicted coordinates
                kpt_loss = F.l1_loss(pred_coords, x1, reduction='none')
                kpt_loss = (kpt_loss.sum(dim=-1) * mask).sum(dim=-1) / normalizer
                kpt_loss_acc = kpt_loss_acc + kpt_loss.mean()

                if t_idx == 0:
                    first_output = output
                    first_x_t = x_t
                    first_similarity_map = similarity_map

            losses['rfm_loss'] = rfm_loss_acc / num_t_samples
            losses['flow_loss'] = flow_loss_acc / num_t_samples
            losses['kpt_loss'] = (kpt_loss_acc / num_t_samples) * 0.5

            # 3. Heatmap loss (if enabled)
            if first_similarity_map is not None and target_q is not None:
                head_losses = self.keypoint_head.get_loss(
                    first_output, first_x_t, first_similarity_map, target_keypoints,
                    target_q, target_weight_q, target_sizes)
                if 'heatmap_loss' in head_losses:
                    losses['heatmap_loss'] = head_losses['heatmap_loss']
            
            # === 4. End-to-End ODE Loss (KEY for train/test consistency) ===
            # Use SAME x0 as flow matching training for stable gradients
            x_ode = x0.clone()  # Start from same noise used in flow matching
            
            # ODE integration (match test steps for consistency)
            num_ode_steps = max(1, int(self.rfm_cfg.num_timesteps))
            dt = 1.0 / num_ode_steps
            ode_step_weight = float(self.train_cfg.get('ode_step_loss_weight', 1.0))
            ode_step_loss_acc = 0.0
            for step in range(num_ode_steps):
                t_ode = torch.full((bs, 1, 1), step * dt, device=img_q.device)
                out_ode, _, _ = self.keypoint_head(
                    feature_q, feature_s, target_s, mask_s,
                    skeleton, all_shots_point_descriptions,
                    coords=x_ode, t=t_ode)
                v_ode = out_ode[-1]
                # No clamp during training to allow gradient flow
                x_ode = x_ode + v_ode * dt

                step_loss = F.l1_loss(x_ode, x1, reduction='none')
                step_loss = (step_loss.sum(dim=-1) * mask).sum(dim=-1) / normalizer
                losses[f'ode_step_loss_{step}'] = step_loss.mean() * ode_step_weight
                ode_step_loss_acc = ode_step_loss_acc + step_loss.mean()

            # Cumulative ODE loss over steps (mean across steps)
            losses['ode_loss'] = ode_step_loss_acc * ode_step_weight
            
            # Accuracy using ODE output (clamp only for evaluation, no gradient)
            keypoint_accuracy = self.keypoint_head.get_accuracy(
                x_ode.detach().clamp(0.0, 1.0),
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

        """Defines the computation performed at every call when testing."""
        batch_size, _, img_height, img_width = img_q.shape

        # Extract features
        mask_s = target_weight_s[0]
        max_points = mask_s.shape[1]
        skeleton = [i['sample_skeleton'][0] for i in img_metas]
        all_shots_point_descriptions = self.extract_text_features(
            img_metas, max_points, mask_s)
        feature_q, feature_s = self.extract_image_features(img_s, img_q)
        
        img_size = torch.tensor([img_width, img_height], device=img_q.device).float()
        
        # === Inference Flow: Support -> Query ===
        # x0 = Centered Support Pose (Prior)
        x = self._get_centered_support_pose(img_metas, img_q.device, img_size, mask_s)
        
        # ODE integration from t=0 to t=1
        num_steps = self.rfm_cfg.num_timesteps if hasattr(self, 'rfm_cfg') else 10
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            # Current time
            t = torch.full((batch_size, 1, 1), step * dt, device=img_q.device)
            
            # Predict velocity field v(x, t) and get heatmap for guidance
            output, _, heatmap = self.keypoint_head(
                feature_q, feature_s, target_s, mask_s,
                skeleton, all_shots_point_descriptions,
                coords=x, t=t)
            
            v_pred = output[-1]  # [B, N, 2] velocity
            
            # === Keyness Guidance (idea.md Step 3) ===
            # v_guided = v + lambda * grad(H(x))
            if heatmap is not None and self.test_cfg.get('use_keyness_guidance', True):
                lambda_guidance = self.test_cfg.get('keyness_lambda', 0.1)
                guidance = keyness_guidance(x, heatmap, lambda_=lambda_guidance)
                v_pred = v_pred + guidance
            
            # Euler step
            x = x + v_pred * dt
        
        predicted_pose = x.detach().cpu().numpy()

        result = {}
        if self.with_keypoint:
            keypoint_result = self.keypoint_head.decode(
                img_metas, predicted_pose, img_size=[img_width, img_height])
            result.update(keypoint_result)


        result.update({
            "points": predicted_pose
        })
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

        # Flow matching: Support -> Query
        img_size = torch.tensor([img_width, img_height], device=img_q.device).float()
        
        # Start from Centered Support Pose
        x0 = self._get_centered_support_pose(img_metas, img_q.device, img_size, mask_s)
        
        if training and target_keypoints is not None:
            t = torch.rand((batch_size, 1, 1), device=img_q.device)
            x1 = target_keypoints / img_size
            x_t = (1 - t) * x0 + t * x1
            coords = x_t
        else:
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
