"""
Paper:
    Tian et al., "DATA: Domain-And-Time Alignment for High-Quality
    Feature Fusion in Collaborative Perception."
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


# =============================================================================
# Gradient Reversal Layer (Ganin & Lempitsky, ICML 2015)
# =============================================================================

class _GradientReversalFn(torch.autograd.Function):


    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class GradientReversalLayer(nn.Module):


    def __init__(self, scale: float = -0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return _GradientReversalFn.apply(x, self.scale)


# =============================================================================
# Observability-constrained Discriminator (CDAM-OD, paper Sec. 3.2.2)
# =============================================================================

class ObservabilityDiscriminator(nn.Module):

    def __init__(self, in_channels: int, grl_scale: float = -0.1):
        super().__init__()
        self.grl = GradientReversalLayer(grl_scale)
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(256, 1, kernel_size=1)
        for layer in (self.conv1, self.conv2):
            nn.init.normal_(layer.weight, std=0.001)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.grl(x)
        x = F.relu(self.conv1(x))
        return self.conv2(x)

class _CentralDifferenceConv(nn.Module):
    """Center weight becomes negative sum of surrounding weights."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

    def get_weight(self):
        w = self.conv.weight
        shape = w.shape
        w_flat = Rearrange('co ci k1 k2 -> co ci (k1 k2)')(w)
        w_cd = w_flat.clone()
        w_cd[:, :, 4] = w_flat[:, :, 4] - w_flat.sum(dim=2)
        w_cd = Rearrange('co ci (k1 k2) -> co ci k1 k2',
                         k1=shape[2], k2=shape[3])(w_cd)
        return w_cd, self.conv.bias


class _AngularDifferenceConv(nn.Module):


    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

    def get_weight(self):
        w = self.conv.weight
        shape = w.shape
        w_flat = Rearrange('co ci k1 k2 -> co ci (k1 k2)')(w)
        w_ad = w_flat - w_flat[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        w_ad = Rearrange('co ci (k1 k2) -> co ci k1 k2',
                         k1=shape[2], k2=shape[3])(w_ad)
        return w_ad, self.conv.bias


class _HorizontalDifferenceConv(nn.Module):


    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=1, bias=True)

    def get_weight(self):
        w = self.conv.weight                       # (Co, Ci, 3)
        co, ci = w.shape[:2]
        w_hd = torch.zeros(co, ci, 9, device=w.device, dtype=w.dtype)
        w_hd[:, :, [0, 3, 6]] = w
        w_hd[:, :, [2, 5, 8]] = -w
        w_hd = Rearrange('co ci (k1 k2) -> co ci k1 k2', k1=3, k2=3)(w_hd)
        return w_hd, self.conv.bias


class _VerticalDifferenceConv(nn.Module):


    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=1, bias=True)

    def get_weight(self):
        w = self.conv.weight                       # (Co, Ci, 3)
        co, ci = w.shape[:2]
        w_vd = torch.zeros(co, ci, 9, device=w.device, dtype=w.dtype)
        w_vd[:, :, [0, 1, 2]] = w
        w_vd[:, :, [6, 7, 8]] = -w
        w_vd = Rearrange('co ci (k1 k2) -> co ci k1 k2', k1=3, k2=3)(w_vd)
        return w_vd, self.conv.bias


class StructConv(nn.Module):


    def __init__(self, dim: int):
        super().__init__()
        # Order matches the legacy DEConv (conv1_1..conv1_5) so any prior
        # state_dict can be migrated by attribute-name rename only.
        self.cd = _CentralDifferenceConv(dim)
        self.hd = _HorizontalDifferenceConv(dim)
        self.vd = _VerticalDifferenceConv(dim)
        self.ad = _AngularDifferenceConv(dim)
        self.vanilla = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

    def forward(self, x):
        w_cd, b_cd = self.cd.get_weight()
        w_hd, b_hd = self.hd.get_weight()
        w_vd, b_vd = self.vd.get_weight()
        w_ad, b_ad = self.ad.get_weight()
        w_v, b_v = self.vanilla.weight, self.vanilla.bias
        w = w_cd + w_hd + w_vd + w_ad + w_v
        b = b_cd + b_hd + b_vd + b_ad + b_v
        return F.conv2d(x, w, b, stride=1, padding=1)


# =============================================================================
# Height-Semantic Verification (IFAM, paper Sec. 3.4, Eq. 14-15)
# =============================================================================

class _SpatialAttention(nn.Module):


    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = x.mean(dim=1, keepdim=True)
        x_max = x.max(dim=1, keepdim=True)[0]
        return self.conv(torch.cat([x_avg, x_max], dim=1))


class _ChannelAttention(nn.Module):


    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=True),
        )

    def forward(self, x):
        return self.mlp(self.gap(x))


class _PixelAttention(nn.Module):


    def __init__(self, dim: int):
        super().__init__()
        self.gconv = nn.Conv2d(2 * dim, dim, 7, padding=3,
                               padding_mode='reflect', groups=dim, bias=True)

    def forward(self, x, init_attn):
        x = x.unsqueeze(2)                          # (B, C, 1, H, W)
        init_attn = init_attn.unsqueeze(2)          # (B, C, 1, H, W)
        cat = torch.cat([x, init_attn], dim=2)      # (B, C, 2, H, W)
        shuffled = Rearrange('b c t h w -> b (c t) h w')(cat)
        return torch.sigmoid(self.gconv(shuffled))


class HeightSemanticVerification(nn.Module):

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.spatial_attn = _SpatialAttention()
        self.channel_attn = _ChannelAttention(dim, reduction)
        self.pixel_attn = _PixelAttention(dim)
        self.proj = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, h_enh, h_fore):
        initial = h_enh + h_fore
        w_init = self.spatial_attn(initial) + self.channel_attn(initial)
        w_verif = torch.sigmoid(self.pixel_attn(initial, w_init))
        out = initial + w_verif * h_enh + (1 - w_verif) * h_fore
        return self.proj(out)


# =============================================================================
# Multi-agent fusion conv (channel concat + 3x3 Conv + BN + ReLU)
# =============================================================================

class AgentFusion(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


# =============================================================================
# IFAM: Instance-focused Feature Aggregation Module
# =============================================================================

def _bias_init_with_prob(prior_prob: float) -> float:

    import math
    return float(-math.log((1 - prior_prob) / prior_prob))


class IFAM(nn.Module):


    def __init__(self, model_cfg: dict):
        super().__init__()
        self.model_cfg = model_cfg
        C = model_cfg.get('in_channels', 384)
        self.align_corners = model_cfg.get('align_corners', False)

        # --- foreground estimator Phi(.) (Supp 1.5) ---
        self.foreground_estimator = nn.Sequential(
            nn.Conv2d(C, C // 2, kernel_size=3, stride=1, padding=1,
                      padding_mode='zeros'),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, 1, kernel_size=1, stride=1, padding=0),
        )
        # Focal-loss style bias init for the final classifier conv.
        self.foreground_estimator[-1].bias.data.fill_(
            _bias_init_with_prob(model_cfg.get('cls_prior_prob', 0.01)))

        # --- IFAM blocks ---
        self.struct_conv = StructConv(C)
        self.verification = HeightSemanticVerification(
            C, reduction=model_cfg.get('verif_reduction', 4))
        self.eps = nn.Parameter(torch.zeros(1))
        self.agent_fusion = AgentFusion(C)

        self.od_discriminator = ObservabilityDiscriminator(
            C + 2, grl_scale=model_cfg.get('grl_scale', -0.1))

        sp_h = model_cfg.get('spatial_map_h', 128)
        sp_w = model_cfg.get('spatial_map_w', 256)
        x_range = model_cfg.get('spatial_map_x_range', (-51.2, 51.2))
        y_range = model_cfg.get('spatial_map_y_range', (-102.4, 102.4))
        xs = torch.linspace(*x_range, steps=sp_h)
        ys = torch.linspace(*y_range, steps=sp_w)
        gx, gy = torch.meshgrid(xs, ys)               # (sp_h, sp_w)
        spatial_map = torch.stack([gy, gx], dim=0)    # (2, sp_h, sp_w)
        spatial_map = spatial_map.abs().unsqueeze(0)  # (1, 2, sp_h, sp_w)
        self.register_buffer('spatial_map', spatial_map)

    def _refine_per_agent(self, feat_in_ego, scores_in_ego):
        """IFAM step 4: per-agent foreground refinement.

        H_refine = 2 * H_verif + (1 - eps) * H_back
        """
        h_fore = feat_in_ego * scores_in_ego
        h_enh = self.struct_conv(feat_in_ego) * scores_in_ego
        h_verif = self.verification(h_enh, h_fore)
        h_back = feat_in_ego * (1 - scores_in_ego)
        return 2 * h_verif + (1 - self.eps) * h_back

    def _observability_align(self, feat_in_ego, scores_in_ego, occ_mask, n_agents):

        if n_agents == 1:
            feat_in_ego = feat_in_ego.repeat(2, 1, 1, 1)
            scores_in_ego = scores_in_ego.repeat(2, 1, 1, 1)
            occ_mask = occ_mask.repeat(2, 1, 1, 1)

        # H_comp_j = (1 - IV_j) * H_i + H_{j->i}; H_{j->i} is already zero
        # outside IV_j (warp_affine_simple zero-fills voids).
        h_comp_j = feat_in_ego[0:1] * (1 - occ_mask[1:2]) + feat_in_ego[1:2]
        h_ego = feat_in_ego[0:1]

        pair = torch.cat([h_comp_j, h_ego], dim=0)               # (2, C, H, W)
        pos = self.spatial_map.repeat(pair.shape[0], 1, 1, 1)    # (2, 2, H, W)
        domain_logits = self.od_discriminator(torch.cat([pair, pos], dim=1))

        # Bundle: 2 logit maps + N M-maps + N IV-maps along the batch dim.
        return torch.cat([domain_logits, scores_in_ego, occ_mask], dim=0)

    def _weighted_fuse(self, feats, scores, record_len, affine_matrix
                       ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        _, _, H, W = feats.shape
        split_feats = regroup(feats, record_len)
        split_scores = regroup(scores, record_len)

        fused_outs: List[torch.Tensor] = []
        domain_outs: List[torch.Tensor] = []
        for b in range(len(record_len)):
            N = record_len[b]
            t_ego = affine_matrix[b][0, :N, :, :]  # ego row of pairwise tfm

            feat_in_ego = warp_affine_simple(
                split_feats[b], t_ego, (H, W), align_corners=self.align_corners)
            scores_in_ego = warp_affine_simple(
                split_scores[b], t_ego, (H, W), align_corners=self.align_corners)
            occ_mask = warp_affine_simple(
                torch.ones_like(scores_in_ego), t_ego, (H, W),
                align_corners=self.align_corners)

            # CDAM-OD (training-time only, via GRL)
            domain_outs.append(self._observability_align(
                feat_in_ego, scores_in_ego, occ_mask, N))

            # IFAM per-agent refinement
            h_refine = self._refine_per_agent(feat_in_ego, scores_in_ego)

            if N == 1:
                h_refine = h_refine.repeat(2, 1, 1, 1)
            ego_collab = torch.cat([h_refine[0:1], h_refine[1:2]], dim=1)
            fused_outs.append(self.agent_fusion(ego_collab))

        return torch.cat(fused_outs, dim=0), domain_outs


    def forward_collab(self, spatial_features, record_len, affine_matrix
                       ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:

        occ_map = self.foreground_estimator(spatial_features)
        # +1e-4 to avoid hard-zero masking in heavily empty regions.
        scores = torch.sigmoid(occ_map) + 1e-4

        fused_feat, domain_set = self._weighted_fuse(
            spatial_features, scores, record_len, affine_matrix)
        return fused_feat, domain_set, [occ_map]