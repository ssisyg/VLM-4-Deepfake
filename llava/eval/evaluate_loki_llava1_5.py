"""
Large Vision-Language Model (VLM) Evaluator for LOKI Deepfake Detection Dataset.
(Kaggle / Jupyter 专用一键运行版)
"""

import os
import gc
import json
import logging
import argparse
from typing import Dict, Tuple
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, LlavaForConditionalGeneration

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# 标准 LLaVA 提示词
PROMPT_TEMPLATE = "USER: <image>\nIs this image a real photograph or a fake AI-generated image? Answer with only one word: real or fake.\nASSISTANT:"

class LOKIVLMDataset(Dataset):
    def __init__(self, json_path: str, img_dir: str):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"🚨 找不到 JSON 文件: {json_path}")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        self.img_dir = img_dir
        
        # 建立全局雷达索引，防止路径不匹配
        logger.info(f"🔍 正在从 {img_dir} 构建全盘图片雷达索引...")
        self.img_index = {}
        for root, _, files in os.walk(img_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.img_index[file] = os.path.join(root, file)
        logger.info(f"✅ 成功索引 {len(self.img_index)} 张图片。")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        item = self.data[idx]
        
        q_type = str(item.get('question_type', '')).lower()
        if "_image_real_" in q_type:
            label = 1  
        elif "_image_fake_" in q_type:
            label = 0  
        else:
            label = 0  
            
        rel_path = item.get('image_path', '')
        img_name = os.path.basename(rel_path)
        actual_img_path = self.img_index.get(img_name, os.path.join(self.img_dir, rel_path))
            
        return actual_img_path, label


def evaluate_llava(model_id: str, dataloader: DataLoader, device: torch.device, output_dir: str):
    logger.info("-" * 60)
    logger.info(f"🚀 正在加载大语言视觉模型: {model_id}")
    
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        # 强制半精度加载，极大节省 Kaggle 显存
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        model.eval()
        logger.info("✅ 模型加载成功！")
    except Exception as e:
        logger.error(f"❌ 加载模型 {model_id} 失败。错误: {e}")
        return

    correct, total_valid = 0, 0
    results_to_save = []
    error_count = 0

    with torch.inference_mode():
        # batch_size=1 是大模型防爆显存的核心
        for batch_paths, batch_labels in tqdm(dataloader, desc="推理中", unit="张"):
            img_path = batch_paths[0]
            true_label = batch_labels[0].item()
            true_str = "real" if true_label == 1 else "fake"
            
            try:
                raw_image = Image.open(img_path).convert("RGB")
            except Exception as e:
                if error_count < 5:
                    logger.warning(f"⚠️ 图片读取失败 [{img_path}]: {e}")
                    error_count += 1
                continue
                
            try:
                inputs = processor(
                    text=PROMPT_TEMPLATE, 
                    images=raw_image, 
                    return_tensors="pt"
                ).to(device, torch.float16)

                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    use_cache=True
                )
                
                # 严谨截断：只获取模型自己的回答部分，去掉 Prompt
                input_len = inputs["input_ids"].shape[1]
                generated_text = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip().lower()
                
            except Exception as e:
                logger.error(f"❌ 推理崩溃 {img_path}. 错误: {e}")
                continue
            
            # 对齐回答
            if "fake" in generated_text and "real" not in generated_text:
                final_pred = "fake"
            elif "real" in generated_text and "fake" not in generated_text:
                final_pred = "real"
            else:
                final_pred = "fake" if "fake" in generated_text else "real"
                
            if final_pred == true_str:
                correct += 1
            total_valid += 1
            
            results_to_save.append({
                "image_path": img_path,
                "true_label": true_str,
                "pred_label": final_pred,
                "raw_model_output": generated_text
            })

            # 🧹 强力垃圾回收：每跑 50 张图清空一次显存，防止 OOM
            del inputs, output_ids, raw_image
            if total_valid % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    if total_valid > 0:
        accuracy = (correct / total_valid) * 100
        logger.info(f"🎉 跑分完成！准确率: {accuracy:.2f}% ({correct}/{total_valid})")
    else:
        logger.warning("跑分完成！但没有处理任何有效图片。")

    safe_model_name = model_id.split('/')[-1]
    output_file = os.path.join(output_dir, f"{safe_model_name}_loki_results.json")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=4, ensure_ascii=False)
        logger.info(f"💾 结果已保存至: {output_file}")
    except Exception as e:
        logger.error(f"❌ 保存结果到硬盘失败: {e}")

# -----------------------------------------------------------------------------
# 运行主程序
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    # Kaggle 默认配置，直接写死默认值
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--data_json", type=str, default="/kaggle/working/LOKI/true_or_false.json")
    parser.add_argument("--img_dir", type=str, default="/kaggle/working/LOKI")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/loki_results/")
    parser.add_argument("--batch_size", type=int, default=1)
    
    # 🔑 核心补丁：屏蔽 Kaggle 的后台参数，直接使用默认值
    return parser.parse_args(args=[])

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        dataset = LOKIVLMDataset(json_path=args.data_json, img_dir=args.img_dir)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    except Exception as e:
        logger.error(f"❌ 数据集初始化失败: {e}")
        return

    evaluate_llava(
        model_id=args.model_id, 
        dataloader=dataloader, 
        device=device, 
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
