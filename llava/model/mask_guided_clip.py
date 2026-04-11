# llava/model/mask_guided_clip.py
# 掩码引导的 CLIP 封装层
# 把 SRM 生成的空间掩码注入 CLIP 的 patch token，引导语义注意力

import torch
import torch.nn as nn


class MaskGuidedCLIP(nn.Module):
    """
    在 CLIPVisionTower 外包一层掩码引导逻辑。

    工作流程：
        1. 原始图像送入冻结的 CLIP → patch tokens (B, N+1, D)
        2. spatial_mask (B, N) 与 patch tokens 做加权强调
        3. 返回引导后的 patch tokens，形状不变

    不修改 CLIP 内部结构，只在输出 token 层操作，安全且轻量。
    """

    def __init__(
        self,
        clip_vision_tower,          # 原始 CLIPVisionTower 实例
        guidance_strength: float = 0.5,   # α，引导强度，可学习
        learnable_alpha: bool = True,
    ):
        super().__init__()
        self.clip = clip_vision_tower   # 保持冻结

        if learnable_alpha:
            # 让模型自己学习最佳引导强度
            self.alpha = nn.Parameter(
                torch.tensor(guidance_strength)
            )
        else:
            self.alpha = guidance_strength

        # 轻量的特征精炼层（可选，让引导后的特征更平滑）
        hidden = clip_vision_tower.hidden_size
        self.refine = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        # 初始化为恒等（不干扰原始特征）
        nn.init.eye_(self.refine[0].weight)
        nn.init.zeros_(self.refine[0].bias)
        nn.init.eye_(self.refine[2].weight)
        nn.init.zeros_(self.refine[2].bias)

    def forward(
        self,
        images: torch.Tensor,
        spatial_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            images:       (B, 3, H, W)
            spatial_mask: (B, N)，来自 SRMEncoder，值域 [0,1]
                          None → 退化为普通 CLIP，方便消融实验

        Returns:
            guided_tokens: (B, N+1, D)，形状与原始 CLIP 输出相同
        """
        # ── Step 1: CLIP 正常前向 ─────────────────────────────
        clip_tokens = self.clip(images)   # (B, N+1, D)
        # clip_tokens[:, 0, :]  → CLS token（全局语义，不动它）
        # clip_tokens[:, 1:, :] → patch tokens（空间局部特征）

        if spatial_mask is None:
            return clip_tokens

        # ── Step 2: 掩码引导 ──────────────────────────────────
        cls_token   = clip_tokens[:, :1, :]    # (B, 1, D)
        patch_tokens = clip_tokens[:, 1:, :]   # (B, N, D)

        # spatial_mask: (B, N) → (B, N, 1) 广播到特征维度
        mask = spatial_mask.unsqueeze(-1)       # (B, N, 1)

        # 强调异常区域：正常区域保持原始权重，异常区域额外增强
        # 公式：token' = token × (1 + α × mask)
        # mask=0 → token' = token（正常保留）
        # mask=1 → token' = token × (1 + α)（异常增强）
        alpha = torch.clamp(self.alpha, 0.0, 2.0)   # 防止梯度爆炸
        guided_patches = patch_tokens * (1.0 + alpha * mask)

        # ── Step 3: 轻量精炼 ──────────────────────────────────
        guided_patches = self.refine(guided_patches)  # (B, N, D)

        # ── Step 4: 拼回 CLS token ────────────────────────────
        guided_tokens = torch.cat([cls_token, guided_patches], dim=1)
        # (B, N+1, D)，形状不变，可以直接送进原有 Projector

        return guided_tokens
