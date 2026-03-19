#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Custom 3D Attention Modules for COSF-Net
#
# This file implements the two core attention modules proposed in:
#   "Enhanced Wild Feline Behavior Recognition via Collaborative SlowFast
#    Network with Heterogeneous Attention Fusion"
#   Submitted to The Visual Computer, Springer Nature.
#
# Modules:
#   - MultiScaleFCALayer3D  : MS-FCA-3D (Multi-Scale Fine-grained Channel
#                             Attention 3D)
#   - DynamicContextBiFormer3D : DC-BiFormer-3D (Dynamic Context Bi-level
#                                Routing Attention 3D)
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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Section 1: Base Components
# =============================================================================

class Mix3D(nn.Module):
    """Learnable adaptive mixing gate for two feature tensors."""
    def __init__(self, m=-0.80):
        super().__init__()
        self.w = nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + \
              fea2 * (1 - mix_factor.expand_as(fea2))
        return out


class FCALayer3D_Base(nn.Module):
    """
    Base FCA layer extended to 3D spatio-temporal features.

    Combines a 1D local convolution branch and a fully-connected global
    branch via cross-correlation, following the FCA design principle
    (Qin et al., ICCV 2021) adapted for volumetric (T x H x W) inputs.

    Args:
        channel (int): Number of input channels.
        reduction (int): Channel reduction ratio for the FC branch.
        b (int): Bias term for adaptive kernel size computation.
        gamma (int): Scale factor for adaptive kernel size computation.
    """
    def __init__(self, channel, reduction=16, b=1, gamma=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k,
                               padding=k // 2, bias=False)

        if reduction > 1:
            self.fc = nn.Sequential(
                nn.Conv3d(channel, channel // reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // reduction, channel, 1)
            )
        else:
            self.fc = nn.Conv3d(channel, channel, 1, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.mix = Mix3D()

    def forward(self, x):
        pooled = self.avg_pool(x)                              # [B,C,1,1,1]

        # Local branch
        x1 = pooled.squeeze(-1).squeeze(-1).squeeze(-1)       # [B,C]
        x1 = x1.unsqueeze(1)                                  # [B,1,C]
        x1 = self.conv1(x1)                                   # [B,1,C]
        x1 = x1.transpose(-1, -2)                             # [B,C,1]

        # Global branch
        x2 = self.fc(pooled).squeeze(-1).squeeze(-1) \
                             .squeeze(-1).unsqueeze(1)         # [B,1,C]

        # Cross-correlation
        out1 = self.sigmoid(torch.sum(
            torch.matmul(x1, x2), dim=1, keepdim=False
        ).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))          # [B,C,1,1,1]

        out2 = self.sigmoid(torch.sum(
            torch.matmul(x2.transpose(-1, -2), x1.transpose(-1, -2)),
            dim=1, keepdim=False
        ).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))          # [B,C,1,1,1]

        out = self.mix(out1, out2)

        # Final refinement
        out = out.squeeze(-1).squeeze(-1).squeeze(-1).unsqueeze(1)
        out = self.conv1(out).transpose(-1, -2)
        out = out.unsqueeze(-1).unsqueeze(-1)
        out = self.sigmoid(out)

        return x * out


# =============================================================================
# Section 2: MS-FCA-3D  (Core Contribution 1)
# =============================================================================

class MultiScaleFCALayer3D(nn.Module):
    """
    Multi-Scale Fine-grained Channel Attention 3D (MS-FCA-3D).

    This module addresses the single-scale limitation of ECA and the lack
    of 3D temporal modeling in the original FCA by:

      1. Building a spatial pyramid at scales {0.5x, 1.0x, 2.0x} while
         keeping the temporal dimension T unchanged, so that temporal
         action coherence is fully preserved.
      2. Assigning a learnable weight omega_k to each scale and normalising
         via Softmax so the network can adapt to species of different body
         sizes and varying camera distances.
      3. Applying a gated residual fusion mechanism:
             Y = G * BN(W_fuse(Z_cat)) + (1 - G) * X
         where the gating factor G is generated from the input via a
         lightweight bottleneck, enabling selective suppression of
         environment noise (wind, vegetation, light flicker) unrelated
         to animal motion.

    Args:
        channel (int): Number of input/output channels.
        reduction (int): Reduction ratio inside each FCA sub-module.
        b, gamma (int): Parameters for adaptive kernel size in FCA base.
        num_scales (int): Number of pyramid scales (default 3).
        use_gate (bool): Whether to use the gated residual mechanism.
    """
    def __init__(self, channel, reduction=16, b=1, gamma=2,
                 num_scales=3, use_gate=True):
        super().__init__()
        self.num_scales = num_scales
        self.use_gate = use_gate

        # One FCA sub-module per scale
        self.fca_modules = nn.ModuleList([
            FCALayer3D_Base(channel, reduction, b, gamma)
            for _ in range(num_scales)
        ])

        # Learnable scale weights (omega_k), Eq. (4) in the paper
        self.scale_weights = nn.Parameter(
            torch.ones(num_scales) / num_scales
        )

        # 1x1x1 fusion convolution W_fuse, Eq. (5)
        self.fusion_conv = nn.Conv3d(
            channel * num_scales, channel, kernel_size=1, bias=True
        )
        self.norm = nn.BatchNorm3d(channel)

        # Gating factor G, Eq. (5)
        if use_gate:
            self.gate = nn.Sequential(
                nn.Conv3d(channel, channel // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // 4, channel, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, x):
        B, C, T, H, W = x.size()

        # Normalised scale weights (Softmax), Eq. (4)
        weights = F.softmax(self.scale_weights, dim=0)

        # Scale factors s_k ∈ {0.5, 1.0, 2.0}
        scale_factors = [0.5, 1.0, 2.0][:self.num_scales]
        outputs = []

        for i, s in enumerate(scale_factors):
            if s == 1.0:
                x_s = x
            else:
                # Scale only spatial (H, W); keep temporal T intact, Eq. (3)
                new_h = max(int(H * s), 1)
                new_w = max(int(W * s), 1)
                x_s = F.interpolate(x, size=(T, new_h, new_w),
                                    mode='trilinear', align_corners=False)

            x_s = self.fca_modules[i](x_s)

            # Restore spatial resolution before concat
            if s != 1.0:
                x_s = F.interpolate(x_s, size=(T, H, W),
                                    mode='trilinear', align_corners=False)

            outputs.append(x_s * weights[i])

        # Concatenate Z_cat ∈ R^{3C×T×H×W}, then fuse, Eq. (5)
        z_cat = torch.cat(outputs, dim=1)
        fused = self.norm(self.fusion_conv(z_cat))

        # Gated residual fusion Y = G ⊙ fused + (1-G) ⊙ X
        if self.use_gate:
            g = self.gate(x)
            return g * fused + (1 - g) * x
        else:
            return x + fused


# =============================================================================
# Section 3: DC-BiFormer-3D  (Core Contribution 2)
# =============================================================================

class DynamicContextBiFormer3D(nn.Module):
    """
    Dynamic Context Bi-level Routing Attention 3D (DC-BiFormer-3D).

    Extends the 2D BiFormer (Zhu et al., CVPR 2023) to volumetric features
    with three targeted improvements for wild-scene video understanding:

      1. Dynamic window predictor: A lightweight bottleneck predicts a scene
         complexity score C ∈ [0,1] and maps it to a routing window size
             W_dyn = W_base + (1-C) * (W_max - W_base)
         so the model uses small windows for complex textures and large
         windows for simple backgrounds, without increasing inference cost.

      2. Adaptive spatio-temporal information fusion: To compensate for the
         global context lost during sparse routing, temporal context V_T and
         spatial context V_S are decoupled via orthogonal pooling:
             V_T = AvgPool_{H,W}(X),  V_S = AvgPool_T(X)
         and gated-fused:
             V_Global = G_ctx ⊙ V_T + (1-G_ctx) ⊙ V_S
         The final output is  Y = BRA_sparse(X, W_dyn) + V_Global.

      3. 3D locally enhanced position encoding (LEPE): The original 2D
         depthwise convolution in BiFormer is replaced by a 3D depthwise
         separable convolution to model temporal position relationships.

    Args:
        dim (int): Input feature dimension (channels).
        num_heads (int): Number of attention heads.
        base_win (int): Minimum (base) routing window size.
        max_win (int): Maximum routing window size.
        topk (int): Top-K regions selected per query in sparse routing.
        qk_scale (float, optional): Manual scale for QK dot product.
        side_dwconv (int): Kernel size for 3D LEPE depthwise conv.
        use_dynamic_window (bool): Enable the dynamic window predictor.
        use_global_context (bool): Enable adaptive spatio-temporal fusion.
    """
    def __init__(self, dim, num_heads=8, base_win=7, max_win=14, topk=4,
                 qk_scale=None, side_dwconv=3,
                 use_dynamic_window=True, use_global_context=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.base_win = base_win
        self.max_win = max_win
        self.topk = topk
        self.use_dynamic_window = use_dynamic_window
        self.use_global_context = use_global_context

        # QKV and output projections
        self.qkv = nn.Conv3d(dim, 3 * dim, kernel_size=1)
        self.proj = nn.Conv3d(dim, dim, kernel_size=1)

        # 3D LEPE (locally enhanced position encoding)
        if side_dwconv > 0:
            self.lepe = nn.Conv3d(
                dim, dim,
                kernel_size=(1, side_dwconv, side_dwconv),
                padding=(0, side_dwconv // 2, side_dwconv // 2),
                groups=dim
            )
        else:
            self.lepe = nn.Identity()

        # Dynamic window predictor (Eq. 6 in the paper)
        if use_dynamic_window:
            self.window_predictor = nn.Sequential(
                nn.Conv3d(dim, dim // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(dim // 4, 1),
                nn.Sigmoid()
            )

        # Adaptive spatio-temporal information fusion (Eqs. 7-11)
        if use_global_context:
            # Temporal context branch: 3×1×1 depthwise conv
            self.temporal_ctx = nn.Sequential(
                nn.AdaptiveAvgPool3d((None, 1, 1)),
                nn.Conv3d(dim, dim,
                          kernel_size=(3, 1, 1), padding=(1, 0, 0),
                          groups=dim),
                nn.ReLU(inplace=True),
                nn.Conv3d(dim, dim, kernel_size=1)
            )
            # Spatial context branch: 1×3×3 depthwise conv
            self.spatial_ctx = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, None, None)),
                nn.Conv3d(dim, dim,
                          kernel_size=(1, 3, 3), padding=(0, 1, 1),
                          groups=dim),
                nn.ReLU(inplace=True),
                nn.Conv3d(dim, dim, kernel_size=1)
            )
            # Context gating unit G_ctx, Eq. (11)
            self.ctx_gate = nn.Sequential(
                nn.Conv3d(dim * 2, dim, kernel_size=1),
                nn.Sigmoid()
            )

    # ------------------------------------------------------------------
    def _get_window_size(self, x):
        """Compute dynamic routing window size W_dyn, Eq. (6)."""
        if not self.use_dynamic_window:
            return self.base_win, self.base_win
        B, C, T, H, W = x.size()
        with torch.no_grad():
            c_score = self.window_predictor(x).mean().item()
        win = int(self.base_win +
                  (self.max_win - self.base_win) * (1 - c_score))
        win_h = min(win, H)
        win_w = min(win, W)
        while win_h > 1 and H % win_h != 0:
            win_h -= 1
        while win_w > 1 and W % win_w != 0:
            win_w -= 1
        return max(1, win_h), max(1, win_w)

    def _global_context(self, v):
        """Compute V_Global via spatio-temporal orthogonal decoupling,
        Eqs. (7)-(11)."""
        if not self.use_global_context:
            return 0
        B, C, T, H, W = v.size()
        v_t = self.temporal_ctx(v).expand(-1, -1, -1, H, W)  # [B,C,T,H,W]
        v_s = self.spatial_ctx(v).expand(-1, -1, T, -1, -1)  # [B,C,T,H,W]
        g = self.ctx_gate(torch.cat([v_t, v_s], dim=1))
        return g * v_t + (1 - g) * v_s

    # ------------------------------------------------------------------
    def forward(self, x):
        B, C, T, H, W = x.size()

        win_h, win_w = self._get_window_size(x)

        # Pad to multiples of window size (spatial only)
        pad_h = (win_h - H % win_h) % win_h
        pad_w = (win_w - W % win_w) % win_w
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, 0))
        _, _, _, Hp, Wp = x.size()

        # QKV projection
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Region-level routing
        n_h, n_w = Hp // win_h, Wp // win_w
        R = n_h * n_w
        q_r = F.adaptive_avg_pool3d(q, (1, n_h, n_w)).view(B, C, R)
        k_r = F.adaptive_avg_pool3d(k, (1, n_h, n_w)).view(B, C, R)
        affinity = torch.matmul(q_r.transpose(-1, -2), k_r) * self.scale
        _, topk_idx = torch.topk(affinity, min(self.topk, R), dim=-1)

        # Token-level sparse attention within windows
        def window_partition(t):
            t = t.view(B, self.num_heads, self.head_dim,
                        T, n_h, win_h, n_w, win_w)
            t = t.permute(0, 1, 4, 6, 3, 5, 7, 2)
            return t.reshape(B, self.num_heads, R,
                              T * win_h * win_w, self.head_dim)

        q_w = window_partition(q)
        k_w = window_partition(k)
        v_w = window_partition(v)

        L = T * win_h * win_w
        K = min(self.topk, R)
        idx = topk_idx.unsqueeze(1).unsqueeze(-1).unsqueeze(-1) \
                      .expand(-1, self.num_heads, -1, -1, L, self.head_dim)

        k_g = torch.gather(
            k_w.unsqueeze(3).expand(-1, -1, -1, K, -1, -1),
            dim=2, index=idx
        ).reshape(B, self.num_heads, R, -1, self.head_dim)

        v_g = torch.gather(
            v_w.unsqueeze(3).expand(-1, -1, -1, K, -1, -1),
            dim=2, index=idx
        ).reshape(B, self.num_heads, R, -1, self.head_dim)

        attn = F.softmax(
            torch.matmul(q_w, k_g.transpose(-1, -2)) * self.scale, dim=-1
        )
        out = torch.matmul(attn, v_g)  # [B, heads, R, L, head_dim]

        # Restore spatial layout
        out = out.view(B, self.num_heads, n_h, n_w,
                       T, win_h, win_w, self.head_dim)
        out = out.permute(0, 1, 7, 4, 2, 5, 3, 6)
        out = out.reshape(B, C, T, Hp, Wp)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :, :H, :W]

        # Add global context V_Global (Eq. 11)
        v_orig = v[:, :, :, :H, :W]
        global_ctx = self._global_context(v_orig)
        if isinstance(global_ctx, torch.Tensor):
            out = out + global_ctx

        # 3D LEPE + output projection
        out = out + self.lepe(v_orig)
        out = self.proj(out)
        return out


# =============================================================================
# Section 4: Backward-compatible aliases
# =============================================================================

class FCALayer3D(MultiScaleFCALayer3D):
    """Drop-in replacement for original FCALayer3D; uses MS-FCA-3D by default."""
    def __init__(self, channel, reduction=16, b=1, gamma=2,
                 num_scales=3, use_gate=True, **kwargs):
        super().__init__(channel, reduction, b, gamma,
                         num_scales=num_scales, use_gate=use_gate)


class BiLevelRoutingAttention3D(DynamicContextBiFormer3D):
    """Drop-in replacement for original BiLevelRoutingAttention3D;
    uses DC-BiFormer-3D by default."""
    def __init__(self, dim, num_heads=8, n_win=7, topk=4,
                 qk_scale=None, side_dwconv=3,
                 use_dynamic_window=True, use_global_context=True, **kwargs):
        base = n_win if isinstance(n_win, int) else n_win[-1]
        super().__init__(dim=dim, num_heads=num_heads,
                         base_win=base, max_win=base * 2,
                         topk=topk, qk_scale=qk_scale,
                         side_dwconv=side_dwconv,
                         use_dynamic_window=use_dynamic_window,
                         use_global_context=use_global_context)


# Keep base-version aliases for reference
BiLevelRoutingAttention3D_Fixed = BiLevelRoutingAttention3D


# =============================================================================
# Section 5: Quick sanity check
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, T, H, W = 2, 64, 8, 56, 56
    x = torch.randn(B, C, T, H, W).to(device)
    print(f"Input : {x.shape}")

    ms_fca = MultiScaleFCALayer3D(C, num_scales=3).to(device)
    y1 = ms_fca(x)
    assert y1.shape == x.shape
    print(f"MS-FCA-3D output : {y1.shape}  [OK]")

    dc_bi = DynamicContextBiFormer3D(C, num_heads=8,
                                     base_win=7, topk=4).to(device)
    y2 = dc_bi(x)
    assert y2.shape == x.shape
    print(f"DC-BiFormer-3D output : {y2.shape}  [OK]")
