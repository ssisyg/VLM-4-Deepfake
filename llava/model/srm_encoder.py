# llava/model/srm_encoder.py
# 微观频率流：SRM 残差 → CNN Encoder → 特征向量 + 空间掩码

import torch
import torch.nn as nn
import torch.nn.functional as F
from llava.srm_utils import apply_srm, srm_to_spatial_mask


class SRMEncoder(nn.Module):
    """
    微观频率流编码器。

    输入：原始图像 (B, 3, H, W)
    输出：
        freq_tokens  : (B, num_freq_tokens, hidden_size)  送 Projector2
        spatial_mask : (B, N_clip_patches)                送 CLIP 引导
    """

    def __init__(
        self,
        out_channels: int = 256,
        hidden_size: int = 1024,    # 和 CLIP ViT-L 对齐
        patch_size: int = 14,       # CLIP patch size
        num_freq_tokens: int = 64,  # 压缩后的频率 token 数量
        threshold_pct: float = 0.65,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_freq_tokens = num_freq_tokens
        self.threshold_pct = threshold_pct

        # ── CNN Encoder：5通道残差图 → 深度特征 ─────────────────
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(5, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # H/2

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # H/4

            # Block 3
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )   # 输出 (B, out_channels, H/4, W/4)

        # ── 掩码生成头（1×1 Conv + Sigmoid）───────────────────
        # 输入是 CNN 中间特征，输出空间掩码
        self.mask_head = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # ── 频率 Token 投影：空间特征 → token 序列 ─────────────
        # 先用自适应池化压缩到固定 token 数，再线性投影
        self.freq_pool = nn.AdaptiveAvgPool2d(
            (int(num_freq_tokens ** 0.5), int(num_freq_tokens ** 0.5))
        )
        self.freq_proj = nn.Linear(out_channels, hidden_size)

    def forward(self, images: torch.Tensor, clip_image_size: int = 336):
        """
        Args:
            images:           (B, 3, H, W)
            clip_image_size:  CLIP 输入分辨率（用于对齐掩码尺寸）

        Returns:
            freq_tokens:  (B, num_freq_tokens, hidden_size)
            spatial_mask: (B, N_patches)  N_patches = (clip_image_size/patch_size)^2
        """
        # ── Step 1: SRM 滤波 → 残差图 ─────────────────────────
        # 先把图像 resize 到 CLIP 的输入尺寸，保证后续对齐
        if images.shape[-1] != clip_image_size:
            images_resized = F.interpolate(
                images, size=(clip_image_size, clip_image_size),
                mode='bilinear', align_corners=False
            )
        else:
            images_resized = images

        srm_residual = apply_srm(images_resized)   # (B, 5, H, W)

        # ── Step 2: CNN 编码 ───────────────────────────────────
        cnn_feat = self.encoder(srm_residual)      # (B, C, H/4, W/4)

        # ── Step 3: 掩码生成 ───────────────────────────────────
        # 3a. CNN 特征图 → 软掩码（空间分辨率 H/4 × W/4）
        mask_spatial = self.mask_head(cnn_feat)    # (B, 1, H/4, W/4)

        # 3b. 上采样到 CLIP patch 分辨率，再展平
        n_patches = clip_image_size // self.patch_size  # 336/14 = 24
        mask_patch = F.interpolate(
            mask_spatial,
            size=(n_patches, n_patches),
            mode='bilinear', align_corners=False
        )                                          # (B, 1, 24, 24)
        spatial_mask = mask_patch.squeeze(1).view(
            mask_patch.shape[0], -1
        )                                          # (B, 576)

        # ── Step 4: 频率 Token 生成 ────────────────────────────
        pooled = self.freq_pool(cnn_feat)          # (B, C, 8, 8)
        B, C, h, w = pooled.shape
        freq_tokens = pooled.view(B, C, h * w).permute(0, 2, 1)
        # (B, 64, C)
        freq_tokens = self.freq_proj(freq_tokens)  # (B, 64, hidden_size)

        return freq_tokens, spatial_mask
