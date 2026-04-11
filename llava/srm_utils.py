# llava/srm_utils.py
import torch
import torch.nn.functional as F
import numpy as np

SRM_KERNELS = torch.tensor([
    # 2nd-order horizontal
    [[[ 0,  0,  0,  0,  0],
      [ 0,  0,  0,  0,  0],
      [ 0, -1,  2, -1,  0],
      [ 0,  0,  0,  0,  0],
      [ 0,  0,  0,  0,  0]]] ,
    # 2nd-order vertical
    [[[ 0,  0,  0,  0,  0],
      [ 0,  0, -1,  0,  0],
      [ 0,  0,  2,  0,  0],
      [ 0,  0, -1,  0,  0],
      [ 0,  0,  0,  0,  0]]],
    # Laplacian
    [[[ 0,  0,  0,  0,  0],
      [ 0,  0,  1,  0,  0],
      [ 0,  1, -4,  1,  0],
      [ 0,  0,  1,  0,  0],
      [ 0,  0,  0,  0,  0]]],
], dtype=torch.float32)  # shape: (3, 1, 5, 5)


def compute_srm_mask(image_tensor: torch.Tensor,
                     num_patches: int = 24,
                     threshold_pct: float = 0.7) -> torch.Tensor:
    """
    image_tensor: (B, C, H, W)，值域 [0,1] 或归一化后的
    num_patches:  CLIP的每边patch数，ViT-L/14@336 → 24
    返回: (B, num_patches*num_patches)  0/1 mask，1=异常区域
    """
    B, C, H, W = image_tensor.shape
    device = image_tensor.device

  
    gray = image_tensor.mean(dim=1, keepdim=True)  # (B,1,H,W)

    kernels = SRM_KERNELS.to(device)  # (3,1,5,5)

    # 逐核卷积，取绝对值后加权求和
    residuals = []
    for i in range(kernels.shape[0]):
        k = kernels[i:i+1]  # (1,1,5,5)
        r = F.conv2d(gray, k, padding=2)  # (B,1,H,W)
        r = r.abs()
        residuals.append(r)

    combined = torch.stack(residuals, dim=0).mean(dim=0)  # (B,1,H,W)

    # 下采样到 patch 分辨率
    mask_spatial = F.interpolate(
        combined,
        size=(num_patches, num_patches),
        mode='bilinear',
        align_corners=False
    ).squeeze(1)  # (B, num_patches, num_patches)

    # 归一化后二值化
    B_size = mask_spatial.shape[0]
    flat = mask_spatial.view(B_size, -1)  # (B, N)
    thresh = torch.quantile(flat, threshold_pct, dim=1, keepdim=True)
    binary = (flat >= thresh).float()  # (B, N)  1=异常

    return binary  # (B, num_patches^2)
