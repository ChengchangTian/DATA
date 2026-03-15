"""DATA: Instance-focused Feature Aggregation Module (IFAM)
with embedded Observability-constrained Discriminator (CDAM-OD).

Paper:
    Tian et al., "DATA: Domain-And-Time Alignment for High-Quality
    Feature Fusion in Collaborative Perception."

This single file implements the fusion stack used by the open-source DATA model:

    StructConv          (Sec. 3.4 / Supp 1.4) -- 5 specialized 3x3 convs
    HeightSemanticVerification (Sec. 3.4, Eq. 14-15) -- attention-gated mixing
    AgentFusion         (multi-agent channel-concat + 1x1 conv)
    ObservabilityDiscriminator (Sec. 3.2.2, Supp 1.1) -- GRL + 2-layer conv
    IFAM                (top-level fusion module, replaces ffnetfuseadv)

PTAM and PHD are not part of this file; see paper Sec. 3.3 and 3.2.1.
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
    """Identity in forward; multiplies gradient by `scale` in backward."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class GradientReversalLayer(nn.Module):
    """Wraps the GRL function so it can be registered as a submodule."""

    def __init__(self, scale: float = -0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return _GradientReversalFn.apply(x, self.scale)


# =============================================================================
# Observability-constrained Discriminator (CDAM-OD, paper Sec. 3.2.2)
# =============================================================================

class ObservabilityDiscriminator(nn.Module):
    """Lightweight conv-based domain discriminator with gradient reversal.

    Architecture (Supp 1.1):
        GRL -> Conv1x1(C -> 256) -> ReLU -> Conv1x1(256 -> 1)

    The input feature is the BEV feature concatenated along the channel axis
    with a 2-channel spatial coordinate map (|x|, |y|), so `in_channels`
    should be (C + 2).
    """

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


# =============================================================================
# StructConv: multi-directional structure enhancement (IFAM, Supp 1.4)
# =============================================================================
#
# Five specialized 3x3 convolutions whose weights are summed at runtime into
# one combined kernel, then applied as a single 3x3 conv. The five branches:
#
#   vanilla : standard 3x3 conv             (feature preservation)
#   cd      : central-difference conv       (center-surround contrast)
#   hd      : horizontal-difference conv    (vertical edges)
#   vd      : vertical-difference conv      (horizontal edges)
#   ad      : angular-difference conv       (diagonal / corner structures)
#
# Each branch stores its own learnable parameters; `get_weight()` materializes
# the equivalent 3x3 kernel from the branch-specific compact representation.
# =============================================================================

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
    """Subtract a permuted (rotated) copy of the kernel."""

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
    """Positive left column, zero center column, negative right column.

    Stored compactly as a Conv1d (kernel length 3) and expanded to a 3x3 2D
    kernel at runtime.
    """

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
    """Positive top row, zero middle row, negative bottom row."""

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
    """IFAM's structural convolution: sum of 5 specialized 3x3 kernels."""

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
    """W_s: channel-wise (avg, max) pool -> 7x7 conv -> 1 channel."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = x.mean(dim=1, keepdim=True)
        x_max = x.max(dim=1, keepdim=True)[0]
        return self.conv(torch.cat([x_avg, x_max], dim=1))


class _ChannelAttention(nn.Module):
    """W_c: GAP -> 1x1 conv reduce -> ReLU -> 1x1 conv expand."""

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
    """W_verif: channel-shuffle([x, W_init]) -> grouped 7x7 conv -> sigmoid.

    The shuffle is implemented by inserting a length-2 axis between x and
    W_init and then flattening it into the channel dim, which interleaves
    their channels (c0_x, c0_init, c1_x, c1_init, ...). The grouped conv
    (groups = dim) then mixes each interleaved pair independently.
    """

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
    """Foreground verification mechanism of IFAM (paper Sec. 3.4).

    Inputs:
        h_enh  -- structurally enhanced foreground feature   (B, C, H, W)
        h_fore -- original foreground feature                (B, C, H, W)

    Computes:
        initial = h_enh + h_fore
        W_init  = W_s + W_c                            (broadcasting)
        W_verif = sigmoid(GConv(CS([initial, W_init])))
        out     = initial + W_verif * h_enh + (1 - W_verif) * h_fore
        return Conv1x1(out)

    Note: the legacy code (CGAFusion in DEA-Net) routes `feat_enh` as `x`
    and `feat_fore` as `y`, so W_verif gates h_enh; this preserves that
    behavior.
    """

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.spatial_attn = _SpatialAttention()
        self.channel_attn = _ChannelAttention(dim, reduction)
        self.pixel_attn = _PixelAttention(dim)
        self.proj = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, h_enh, h_fore):
        initial = h_enh + h_fore
        w_init = self.spatial_attn(initial) + self.channel_attn(initial)
        # NOTE: `pixel_attn` already applies sigmoid internally, but the
        # legacy CGAFusion (from DEA-Net) applies sigmoid a SECOND time on
        # its output. This double-sigmoid squashes w_verif into roughly
        # [sigmoid(0), sigmoid(1)] = [0.500, 0.731] rather than [0, 1].
        # The trained `pixel_attn.gconv` weights were optimised under this
        # soft-gating regime, so we preserve it here for exact numerical
        # compatibility with checkpoints trained on the legacy code.
        w_verif = torch.sigmoid(self.pixel_attn(initial, w_init))
        out = initial + w_verif * h_enh + (1 - w_verif) * h_fore
        return self.proj(out)


# =============================================================================
# Multi-agent fusion conv (channel concat + 3x3 Conv + BN + ReLU)
# =============================================================================

class AgentFusion(nn.Module):
    """Pairwise (ego, collab) fusion: channel-concat, then project back to C."""

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
    """Bias init for the foreground classifier head (focal-loss style)."""
    import math
    return float(-math.log((1 - prior_prob) / prior_prob))


class IFAM(nn.Module):
    """Instance-focused Feature Aggregation Module (paper Sec. 3.4).

    Pipeline (per batch):
        1. Warp every agent's BEV feature into the ego frame.
        2. Predict a foreground / observability map M via Phi (foreground_estimator).
        3. (CDAM-OD branch, training-only effect via GRL) -- void-filled
           discriminator on the common observable area; produces a domain logit
           map plus the supervision tensors (M, IV) bundled together for the
           downstream BCE loss.
        4. (IFAM main path)
                H_fore   = H * M
                H_enh    = StructConv(H) * M
                H_verif  = HeightSemanticVerification(H_enh, H_fore)
                H_back   = H * (1 - M)
                H_refine = 2 * H_verif + (1 - eps) * H_back
        5. Pad single-agent batches to 2 (ego twice), then concat ego and
           collab along channel and run AgentFusion to get the fused BEV.

    Config keys (`model_cfg`):
        in_channels        : int   (default 384)
        align_corners      : bool  (default False)
        verif_reduction    : int   (default 4)
        spatial_map_h      : int   (default 128)
        spatial_map_w      : int   (default 256)
        spatial_map_x_range: (float, float) (default (-51.2, 51.2))
        spatial_map_y_range: (float, float) (default (-102.4, 102.4))
        grl_scale          : float (default -0.1)
        cls_prior_prob     : float (default 0.01)
    """

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
        # eps: zero-initialised learnable scalar that balances the background
        # term (paper denotes this `epsilon`; legacy code called it `gamma`).
        self.eps = nn.Parameter(torch.zeros(1))
        self.agent_fusion = AgentFusion(C)

        # --- CDAM-OD discriminator (input has 2 extra spatial-coord channels) ---
        self.od_discriminator = ObservabilityDiscriminator(
            C + 2, grl_scale=model_cfg.get('grl_scale', -0.1))

        # --- positional / spatial map for OD (BEV xy magnitudes) ---
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

    # ------------------------------------------------------------------ helpers

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
        """CDAM-OD: void-filled adversarial domain alignment (paper Eq. 4-6).

        For a single-agent batch (n_agents == 1) we duplicate the ego
        feature so the discriminator still receives a (collab, ego) pair;
        the resulting logits are not used by the loss in this regime, but
        keeping the call shapes uniform avoids branching in the trainer.

        Returns a tensor that bundles (domain_logits, M, IV) along dim 0
        for the downstream BCE loss to slice.
        """
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
        """Run OD + IFAM refinement per batch sample, then AgentFusion."""
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

            # Pad N==1 -> 2 agents (ego twice) so AgentFusion's 2C input is valid
            if N == 1:
                h_refine = h_refine.repeat(2, 1, 1, 1)
            ego_collab = torch.cat([h_refine[0:1], h_refine[1:2]], dim=1)
            fused_outs.append(self.agent_fusion(ego_collab))

        return torch.cat(fused_outs, dim=0), domain_outs

    # ------------------------------------------------------------------ public

    def forward_collab(self, spatial_features, record_len, affine_matrix
                       ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Run the full IFAM (+ CDAM-OD signal) pass.

        Args:
            spatial_features : (sum(record_len), C, H, W) per-agent BEV features.
            record_len       : list[int], number of agents per sample.
            affine_matrix    : (B, L, L, 2, 3) normalised pairwise transforms.

        Returns:
            fused_feat  : (B, C, H, W) ego-frame fused BEV feature.
            domain_set  : list of CDAM-OD bundles (one per sample) for loss.
            occ_maps    : list with the foreground occupancy logits (one entry,
                          shape (sum(record_len), 1, H, W)).
        """
        occ_map = self.foreground_estimator(spatial_features)
        # +1e-4 to avoid hard-zero masking in heavily empty regions.
        scores = torch.sigmoid(occ_map) + 1e-4

        fused_feat, domain_set = self._weighted_fuse(
            spatial_features, scores, record_len, affine_matrix)
        return fused_feat, domain_set, [occ_map]