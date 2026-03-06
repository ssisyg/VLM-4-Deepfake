import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SRMConv2d(nn.Module):
    def __init__(self, inc=3):
        super(SRMConv2d, self).__init__()
        self.trk_weights = self._build_srm_filters(inc)
        
    def _build_srm_filters(self, inc):
        # 这里定义 SRM 的高频滤波矩阵 (通常是 3 个基础滤波器)
        # q1, q2, q3 等常规 SRM 核
        # 为简化示例，这里用最经典的 5x5 SRM 滤波核张量代替
        # 你可以根据你已有的代码填入精确的数值
        weight = torch.randn(3, 1, 5, 5) # 示例占位，假设输出 3 个通道的噪声图
        weight = weight.repeat(1, inc, 1, 1)
        return nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        return F.conv2d(x, self.trk_weights, stride=1, padding=2)

class SRMVisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. SRM 滤波器 (必须冻结)
        self.srm_filter = SRMConv2d(inc=3)
        self.srm_filter.requires_grad_(False)
        
        # 2. ResNet18 骨干 (解冻，让它学习噪声)
        resnet = models.resnet18(pretrained=True)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.hidden_size = 512
        # 注意：这里我们不写 for param in self.parameters(): param.requires_grad = False
        # 默认让 cnn_backbone 保持 requires_grad = True

    def forward(self, x):
        noise = self.srm_filter(x)
        features = self.cnn_backbone(noise)
        # 将 [B, 512, H, W] 展平为 [B, N, 512] 的序列形式，对齐 CLIP 特征
        features = features.flatten(2).transpose(1, 2)
        return features

class SRMProjector(nn.Module):
    def __init__(self, srm_hidden_size=512, llm_hidden_size=4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(srm_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

    def forward(self, x):
        return self.proj(x)