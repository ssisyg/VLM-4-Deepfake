#!/bin/bash
# =========================================================================
# 🛠️ LLaVA-SRM-Deepfake Environment Setup
# =========================================================================

echo "🚀 [1/3] Setting up workspace..."
cd /kaggle/working
cp -r /kaggle/input/my-llava-srm-code/LLaVA /kaggle/working/
cd /kaggle/working/LLaVA

echo "📦 [2/3] Upgrading pip and installing core dependencies..."
pip install --upgrade pip
pip install transformers==4.37.2 accelerate==0.24.1 peft==0.4.0 sentencepiece einops==0.6.1 shortuuid==1.0.11 httpx==0.24.0 markdown2==2.4.10 requests pillow

echo "🔧 [3/3] Installing bitsandbytes and LLaVA package (No-Deps Mode)..."
pip install --upgrade bitsandbytes
pip install --no-deps -e .

echo "✅ Environment setup completed successfully!"
