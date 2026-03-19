#!/usr/bin/env python3
# =============================================================================
# COSF-Net: Collaborative SlowFast Network with Heterogeneous Attention Fusion
#
# This file implements the full COSF-Net architecture proposed in:
#   "Enhanced Wild Feline Behavior Recognition via Collaborative SlowFast
#    Network with Heterogeneous Attention Fusion"
#   Submitted to The Visual Computer, Springer Nature.
#
# Key class:
#   SlowFastDualAttention  —  the COSF-Net model registered for the
#                             SlowFast framework.
#
# If you use this code, please cite the above paper.
#
# License: CC BY-NC 4.0
#   https://creativecommons.org/licenses/by-nc/4.0/
#   Free for academic/non-commercial use with attribution.
#   Commercial use is prohibited without explicit written permission.
#
# Copyright (c) 2026 Yawei Ji, Yaqin Zhao, Xu Lu, Xiangna Sun
#   Nanjing Forestry University
# =============================================================================

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

import SlowFast.slowfast.utils.weight_init_helper as init_helper
from SlowFast.slowfast.models.batchnorm_helper import get_norm
from SlowFast.slowfast.models import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY

# Core attention modules (this repository)
from .custom_attention_3d import FCALayer3D, DynamicContextBiFormer3D

# ---------------------------------------------------------------------------
# Static configuration tables (unchanged from original SlowFast codebase)
# ---------------------------------------------------------------------------
_MODEL_STAGE_DEPTH = {
    18: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
}
_TEMPORAL_KERNEL_BASIS = {
    "slowfast": [[[1], [5]], [[1], [3]], [[1], [3]], [[3], [3]], [[3], [3]]]
}
_POOL1 = {"slowfast": [[1, 1, 1], [1, 1, 1]]}


# =============================================================================
# Heterogeneous Attention Feature Fusion Module (HAFF)
# =============================================================================

class FuseFastAndSlow(nn.Module):
    """
    Heterogeneous Attention Feature Fusion (HAFF) module.

    This module is inserted after each residual stage (conv1, res2, res3,
    res4) of the dual-pathway network to enable bidirectional, deep
    interaction between the Fast-path motion features and the Slow-path
    semantic features.

    Fast → Slow direction:
        MS-FCA-3D (MultiScaleFCALayer3D) performs multi-scale channel
        attention on the temporally downsampled Fast features, achieving
        scale adaptation and environmental noise suppression.

    Slow → Fast direction:
        DC-BiFormer-3D (DynamicContextBiFormer3D) performs sparse spatial
        attention on the channel-compressed Slow features, realising
        adaptive receptive field adjustment and global context compensation.

    The enhanced features are concatenated with the original pathway
    features as the output.

    Args:
        dim_in (list[int]): [dim_slow, dim_fast] channel counts.
        norm_module: Normalisation layer constructor (e.g. BatchNorm3d).
        cfg: Global config node.
        window_size: Per-stage routing window size (int or list).
    """

    def __init__(self, dim_in, norm_module, cfg, window_size=None):
        super().__init__()
        dim_slow, dim_fast = dim_in

        # ------------------------------------------------------------------
        # Fast → Slow: MS-FCA-3D
        # ------------------------------------------------------------------
        self.downsample_t_fast = nn.MaxPool3d(
            kernel_size=(cfg.SLOWFAST.ALPHA, 1, 1),
            stride=(cfg.SLOWFAST.ALPHA, 1, 1)
        )

        use_multiscale = getattr(cfg.SLOWFAST, "USE_MULTISCALE_FCA", True)
        num_scales = getattr(cfg.SLOWFAST, "FCA_NUM_SCALES", 3) \
            if use_multiscale else 1

        self.attn_f2s = FCALayer3D(
            channel=dim_fast,
            reduction=getattr(cfg.SLOWFAST, "ATTENTION_REDUCTION", 16),
            b=getattr(cfg.SLOWFAST, "FCA_B", 1),
            gamma=getattr(cfg.SLOWFAST, "FCA_GAMMA", 2),
            num_scales=num_scales,
            use_gate=getattr(cfg.SLOWFAST, "FCA_USE_GATE", True)
        )
        self.bn_f2s = norm_module(dim_fast)
        self.relu_f2s = nn.ReLU(inplace=True)

        # ------------------------------------------------------------------
        # Slow → Fast: DC-BiFormer-3D
        # ------------------------------------------------------------------
        self.downsample_c_slow = nn.Conv3d(
            dim_slow, dim_slow // cfg.SLOWFAST.BETA_INV,
            kernel_size=1, bias=False
        )
        self.bn_slow = norm_module(dim_slow // cfg.SLOWFAST.BETA_INV)
        self.relu_slow = nn.ReLU(inplace=True)

        base_win = getattr(cfg.SLOWFAST, "BASE_WINDOW_SIZE", 7)
        max_win = getattr(cfg.SLOWFAST, "MAX_WINDOW_SIZE", 14)

        self.attn_s2f = DynamicContextBiFormer3D(
            dim=dim_slow // cfg.SLOWFAST.BETA_INV,
            num_heads=getattr(cfg.SLOWFAST, "ATTENTION_NUM_HEADS", 8),
            base_win=base_win,
            max_win=max_win,
            topk=getattr(cfg.SLOWFAST, "ATTENTION_TOPK", 4),
            side_dwconv=3,
            use_dynamic_window=getattr(
                cfg.SLOWFAST, "USE_DYNAMIC_WINDOW", True),
            use_global_context=getattr(
                cfg.SLOWFAST, "USE_GLOBAL_CONTEXT", True)
        )
        self.bn_s2f = norm_module(dim_slow // cfg.SLOWFAST.BETA_INV)
        self.relu_s2f = nn.ReLU(inplace=True)
        self.upsample_s2f = nn.Upsample(
            scale_factor=(cfg.SLOWFAST.ALPHA, 1, 1), mode="nearest"
        )

    def forward(self, x):
        x_slow, x_fast = x[0], x[1]

        # Fast → Slow branch
        fast_branch = self.downsample_t_fast(x_fast)
        fast_branch = self.relu_f2s(self.bn_f2s(self.attn_f2s(fast_branch)))

        # Slow → Fast branch
        slow_branch = self.relu_slow(
            self.bn_slow(self.downsample_c_slow(x_slow))
        )
        slow_branch = self.relu_s2f(self.bn_s2f(self.attn_s2f(slow_branch)))
        slow_branch = self.upsample_s2f(slow_branch)

        # Concatenate enhanced features with original pathway features
        x_slow_out = torch.cat([x_slow, fast_branch], dim=1)
        x_fast_out = torch.cat([x_fast, slow_branch], dim=1)

        return [x_slow_out, x_fast_out]


# =============================================================================
# COSF-Net  (SlowFastDualAttention)
# =============================================================================

@MODEL_REGISTRY.register()
class SlowFastDualAttention(nn.Module):
    """
    COSF-Net: Collaborative SlowFast Network with Heterogeneous Attention
    Fusion for wild feline behavior recognition.

    Built on top of Dual-SlowFast (Wei et al., CVIU 2022) with the ECA and
    SA modules replaced by the proposed MS-FCA-3D and DC-BiFormer-3D,
    respectively, embedded inside the HAFF module (FuseFastAndSlow).

    The network follows the standard SlowFast two-pathway design:
        Input → s1 → HAFF → s2 → HAFF → pool → s3 → HAFF
              → s4 → HAFF → s5 → Head
    """

    def __init__(self, cfg):
        super().__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2

        self._validate_config(cfg)
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    # ------------------------------------------------------------------
    def _validate_config(self, cfg):
        assert cfg.MODEL.ARCH in {"slowfast"}, \
            f"Unsupported architecture: {cfg.MODEL.ARCH}"
        assert cfg.RESNET.DEPTH in {50, 101, 152}, \
            f"Unsupported ResNet depth: {cfg.RESNET.DEPTH}"

        window_sizes = getattr(cfg.SLOWFAST, "ATTENTION_WINDOW_SIZE", 7)
        topk = getattr(cfg.SLOWFAST, "ATTENTION_TOPK", 4)

        if isinstance(window_sizes, list) and len(window_sizes) == 4:
            for i, ws in enumerate(window_sizes):
                if isinstance(ws, (list, tuple)) and len(ws) == 3:
                    min_sizes = [56, 56, 28, 14]
                    n_h = (min_sizes[i] + ws[1] - 1) // ws[1]
                    n_w = (min_sizes[i] + ws[2] - 1) // ws[2]
                    if topk > n_h * n_w:
                        warnings.warn(
                            f"Stage {i}: topk({topk}) exceeds the number of "
                            f"routing regions ({n_h * n_w}). Consider "
                            f"reducing topk or increasing window size."
                        )

    # ------------------------------------------------------------------
    def _construct_network(self, cfg):
        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]
        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]
        pool_size = _POOL1[cfg.MODEL.ARCH]

        window_sizes = getattr(cfg.SLOWFAST, "ATTENTION_WINDOW_SIZE", 7)
        stage_windows = window_sizes[:4] if (
            isinstance(window_sizes, list) and len(window_sizes) >= 4
        ) else [[4, 14, 14], [4, 7, 7], [2, 7, 7], [2, 7, 7]]

        # ---- Stage 1: Stem ----
        s1_dim_slow = width_per_group
        s1_dim_fast = width_per_group // cfg.SLOWFAST.BETA_INV

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[s1_dim_slow, s1_dim_fast],
            kernel=[temp_kernel[0][0] + [7, 7],
                    temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[[temp_kernel[0][0][0] // 2, 3, 3],
                     [temp_kernel[0][1][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )
        self.s1_fuse = FuseFastAndSlow(
            [s1_dim_slow, s1_dim_fast],
            self.norm_module, cfg, window_size=stage_windows[0]
        )

        # ---- Stage 2 ----
        s2_dim_slow = width_per_group * 4
        s2_dim_fast = s2_dim_slow // cfg.SLOWFAST.BETA_INV

        self.s2 = resnet_helper.ResStage(
            dim_in=[s1_dim_slow + s1_dim_fast,
                    s1_dim_fast + s1_dim_fast],
            dim_out=[s2_dim_slow, s2_dim_fast],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2, num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = FuseFastAndSlow(
            [s2_dim_slow, s2_dim_fast],
            self.norm_module, cfg, window_size=stage_windows[1]
        )

        for pathway in range(self.num_pathways):
            self.add_module(
                f"pathway{pathway}_pool",
                nn.MaxPool3d(kernel_size=pool_size[pathway],
                             stride=pool_size[pathway],
                             padding=[0, 0, 0])
            )

        # ---- Stage 3 ----
        s3_dim_slow = width_per_group * 8
        s3_dim_fast = s3_dim_slow // cfg.SLOWFAST.BETA_INV

        self.s3 = resnet_helper.ResStage(
            dim_in=[s2_dim_slow + s2_dim_fast,
                    s2_dim_fast + s2_dim_fast],
            dim_out=[s3_dim_slow, s3_dim_fast],
            dim_inner=[dim_inner * 2,
                       dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2, num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastAndSlow(
            [s3_dim_slow, s3_dim_fast],
            self.norm_module, cfg, window_size=stage_windows[2]
        )

        # ---- Stage 4 ----
        s4_dim_slow = width_per_group * 16
        s4_dim_fast = s4_dim_slow // cfg.SLOWFAST.BETA_INV

        self.s4 = resnet_helper.ResStage(
            dim_in=[s3_dim_slow + s3_dim_fast,
                    s3_dim_fast + s3_dim_fast],
            dim_out=[s4_dim_slow, s4_dim_fast],
            dim_inner=[dim_inner * 4,
                       dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2, num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastAndSlow(
            [s4_dim_slow, s4_dim_fast],
            self.norm_module, cfg, window_size=stage_windows[3]
        )

        # ---- Stage 5 ----
        s5_dim_slow = width_per_group * 32
        s5_dim_fast = s5_dim_slow // cfg.SLOWFAST.BETA_INV

        self.s5 = resnet_helper.ResStage(
            dim_in=[s4_dim_slow + s4_dim_fast,
                    s4_dim_fast + s4_dim_fast],
            dim_out=[s5_dim_slow, s5_dim_fast],
            dim_inner=[dim_inner * 8,
                       dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2, num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        # ---- Classification Head ----
        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[s5_dim_slow, s5_dim_fast],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[
                    [cfg.DATA.NUM_FRAMES // cfg.SLOWFAST.ALPHA, 1, 1],
                    [cfg.DATA.NUM_FRAMES, 1, 1],
                ],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2,
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[s5_dim_slow, s5_dim_fast],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=(
                    [None, None] if cfg.MULTIGRID.SHORT_CYCLE
                    else [
                        [cfg.DATA.NUM_FRAMES // cfg.SLOWFAST.ALPHA,
                         cfg.DATA.TRAIN_CROP_SIZE // 32,
                         cfg.DATA.TRAIN_CROP_SIZE // 32],
                        [cfg.DATA.NUM_FRAMES,
                         cfg.DATA.TRAIN_CROP_SIZE // 32,
                         cfg.DATA.TRAIN_CROP_SIZE // 32],
                    ]
                ),
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

    # ------------------------------------------------------------------
    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)

        for pathway in range(self.num_pathways):
            pool = getattr(self, f"pathway{pathway}_pool")
            x[pathway] = pool(x[pathway])

        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)

        if self.enable_detection:
            return self.head(x, bboxes)
        return self.head(x)
