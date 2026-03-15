"""
Paper:
    Tian et al., "DATA: Domain-And-Time Alignment for High-Quality
    Feature Fusion in Collaborative Perception."
"""
import importlib

import torch
import torch.nn as nn

from opencood.models.fuse_modules.data_fusion import IFAM
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.utils.model_utils import check_trainable_module
from opencood.utils.transformation_utils import normalize_pairwise_tfm

class SEBlock(nn.Module):
    """Squeeze-and-excitation refinement: global gate per channel."""

    def __init__(self, channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attn(x)

def _resolve_encoder_class(core_method: str):
    """Look up the encoder class in opencood.models.heter_encoders by name."""
    mod = importlib.import_module('opencood.models.heter_encoders')
    target = core_method.replace('_', '').lower()
    for name, cls in mod.__dict__.items():
        if name.lower() == target:
            return cls
    raise ValueError(f"No encoder class matches core_method={core_method!r}")


class DATA(nn.Module):

    def __init__(self, args: dict):
        super().__init__()
        self.args = args

        modalities = [k for k in args.keys() if k.startswith('m') and k[1:].isdigit()]
        if len(modalities) != 1:
            raise ValueError(
                f"DATA expects exactly one modality config; got {modalities}")
        m = modalities[0]
        self.modality_name = m

        cav_range = args['lidar_range']
        self.H = cav_range[4] - cav_range[1]
        self.W = cav_range[3] - cav_range[0]
        self.fake_voxel_size = 1

        cfg = args[m]
        EncoderCls = _resolve_encoder_class(cfg['core_method'])
        setattr(self, f'encoder_{m}', EncoderCls(cfg['encoder_args']))
        setattr(self, f'backbone_{m}', BaseBEVBackbone(
            cfg['backbone_args'], cfg['backbone_args'].get('inplanes', 64)))
        setattr(self, f'aligner_{m}', AlignNet(cfg['aligner_args']))

        self.ifam = IFAM(args['fusion_backbone'])

        self.se_block = SEBlock(args['in_head'])

        self.shrink_flag = 'shrink_header' in args
        if self.shrink_flag:
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        c, n_anchor = args['in_head'], args['anchor_number']
        n_bins = args['dir_args']['num_bins']
        self.cls_head = nn.Conv2d(c, n_anchor, kernel_size=1)
        self.reg_head = nn.Conv2d(c, 7 * n_anchor, kernel_size=1)
        self.dir_head = nn.Conv2d(c, n_bins * n_anchor, kernel_size=1)

        self.compress = 'compressor' in args
        if self.compress:
            self.compressor = NaiveCompressor(args['compressor']['input_dim'],
                                              args['compressor']['compress_ratio'])

        self._configure_trainable()
        check_trainable_module(self)

    def _configure_trainable(self):
        if not self.compress:
            return
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.compressor.train()
        for p in self.compressor.parameters():
            p.requires_grad_(True)

    def forward(self, data_dict):
        m = self.modality_name
        record_len = data_dict['record_len']
        affine = normalize_pairwise_tfm(
            data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)

        feat = getattr(self, f'encoder_{m}')(data_dict, m)
        feat = getattr(self, f'backbone_{m}')({'spatial_features': feat})['spatial_features_2d']
        feat = getattr(self, f'aligner_{m}')(feat)

        if self.compress:
            feat = self.compressor(feat)

        fused, domain_set, occ_maps = self.ifam.forward_collab(
            feat, record_len, affine)

        fused = self.se_block(fused)
        if self.shrink_flag:
            fused = self.shrink_conv(fused)

        return {
            'pyramid': 'collab',
            'cls_preds': self.cls_head(fused),
            'reg_preds': self.reg_head(fused),
            'dir_preds': self.dir_head(fused),
            'occ_single_list': occ_maps,
            'domain': domain_set,
        }