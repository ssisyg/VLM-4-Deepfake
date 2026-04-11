# llava/srm_utils.py
# SRM 滤波器工具 —— 固定权重，不参与训练

import torch
import torch.nn.functional as F


# ── 5 个代表性 SRM 核（完整版可扩展到 30 个）──────────────────
# 每个 shape: (1, 1, 5, 5)，对灰度图做卷积
_SRM_KERNELS_RAW = [
    # 1. 二阶水平残差
    [[ 0,  0,  0,  0,  0],
     [ 0,  0,  0,  0,  0],
     [ 0, -1,  2, -1,  0],
     [ 0,  0,  0,  0,  0],
     [ 0,  0,  0,  0,  0]],
    # 2. 二阶垂直残差
    [[ 0,  0,  0,  0,  0],
     [ 0,  0, -1,  0,  0],
     [ 0,  0,  2,  0,  0],
     [ 0,  0, -1,  0,  0],
     [ 0,  0,  0,  0,  0]],
    # 3. 对角线残差
    [[ 0,  0,  0,  0,  0],
     [ 0, -1,  0,  1,  0],
     [ 0,  0,  0,  0,  0],
     [ 0,  1,  0, -1,  0],
     [ 0,  0,  0,  0,  0]],
    # 4. 拉普拉斯（各向同性）
    [[ 0,  0,  0,  0,  0],
     [ 0,  0,  1,  0,  0],
     [ 0,  1, -4,  1,  0],
     [ 0,  0,  1,  0,  0],
     [ 0,  0,  0,  0,  0]],
    # 5. 全方向高阶
    [[-1,  2, -2,  2, -1],
     [ 2, -6,  8, -6,  2],
     [-2,  8,-12,  8, -2],
     [ 2, -6,  8, -6,  2],
     [-1,  2, -2,  2, -1]],
]

# 构建 (5, 1, 5, 5) 的固定权重张量
SRM_WEIGHT = torch.tensor(
    _SRM_KERNELS_RAW, dtype=torch.float32
).unsqueeze(1) / 12.0   # 归一化幅度


def apply_srm(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    对输入图像应用 SRM 滤波器，返回残差特征图。

    Args:
        image_tensor: (B, C, H, W)，值域任意（会先转灰度）

    Returns:
        residual: (B, 5, H, W)，每通道对应一个 SRM 核的绝对残差
    """
    device = image_tensor.device
    kernels = SRM_WEIGHT.to(device)              # (5, 1, 5, 5)

    # 转灰度
    gray = image_tensor.mean(dim=1, keepdim=True)  # (B, 1, H, W)

    # 一次性卷积，得 (B, 5, H, W)
    residual = F.conv2d(gray, kernels, padding=2)
    residual = residual.abs()

    # 逐样本归一化到 [0, 1]
    B = residual.shape[0]
    flat = residual.view(B, 5, -1)
    min_v = flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
    max_v = flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    residual = (residual - min_v) / (max_v - min_v + 1e-8)

    return residual  # (B, 5, H, W)


def srm_to_spatial_mask(
    residual: torch.Tensor,
    patch_size: int = 14,
    threshold_pct: float = 0.65,
    soft: bool = True,
) -> torch.Tensor:
    """
    将 SRM 残差图转换为与 CLIP patch token 对齐的空间掩码。

    Args:
        residual:       (B, 5, H, W)  apply_srm 的输出
        patch_size:     CLIP 的 patch 大小（ViT-L/14 → 14）
        threshold_pct:  高于该分位数的区域标为异常（0~1）
        soft:           True → 软掩码（连续值），False → 二值掩码

    Returns:
        mask: (B, N)，N = (H/patch_size) * (W/patch_size)
              值域 [0,1]，1 = 异常区域
    """
    B, _, H, W = residual.shape

    # 5 个通道加权融合
    weights = torch.tensor([0.15, 0.15, 0.20, 0.25, 0.25],
                            device=residual.device).view(1, 5, 1, 1)
    combined = (residual * weights).sum(dim=1, keepdim=True)  # (B,1,H,W)

    # 下采样到 patch 分辨率
    n_h = H // patch_size
    n_w = W // patch_size
    patch_map = F.avg_pool2d(combined, kernel_size=patch_size)  # (B,1,n_h,n_w)
    patch_map = patch_map.squeeze(1)                             # (B, n_h, n_w)

    # 归一化
    flat = patch_map.view(B, -1)  # (B, N)
    min_v = flat.min(dim=1, keepdim=True)[0]
    max_v = flat.max(dim=1, keepdim=True)[0]
    flat = (flat - min_v) / (max_v - min_v + 1e-8)

    if soft:
        return flat  # 软掩码，直接返回连续值

    # 二值掩码：top-(1-threshold_pct) 的区域设为 1
    thresh = torch.quantile(flat, threshold_pct, dim=1, keepdim=True)
    binary = (flat >= thresh).float()
    return binary  # (B, N)
