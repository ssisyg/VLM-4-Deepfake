  # 🕵️‍♂️ View2Verify: A Dual-Stream Framework for Deepfake Detection.（基于VLM的深度伪造图像检测模型）

本项目为本科毕业设计代码仓库。基于大型视觉语言模型 (VLM) LLaVA 架构，通过引入 SRM（Spatial Rich Model）等特定微调策略，赋予模型精准识别 Deepfake（AI生成/篡改图片）的能力。

## ✨ 项目亮点
* **基础架构**：Vicuna-7B + LLaVA Vision Encoder
* **核心创新**：利用 LoRA 进行参数高效微调 (PEFT)，专门针对 AI 生成图像的伪影、光影不一致、边缘瑕疵进行特征提取。
* **轻量化**：仅需加载核心 LoRA 权重与 Vision Projector 即可完成推理。

## 📁 核心文件说明
* `predict.py`：核心推理脚本，包含模型的手动组装与图片预测逻辑。
* `cog.yaml` / `pyproject.toml`：项目环境依赖配置。

## 🚀 快速开始 (Quick Start)

### 1. 环境准备
确保你的环境中已安装 PyTorch，然后安装相关依赖：
```bash
pip install -r requirements.txt
