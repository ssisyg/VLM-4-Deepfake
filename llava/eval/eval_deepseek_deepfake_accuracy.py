"""
DeepSeek Janus-Pro (7B) Deepfake Detection Baseline 
包含单卡强锁 (解决 cuda:0/1 冲突) + Meta Tensor 补丁 + 数据类型对齐。
"""

import os
import sys
import transformers 

# ==========================================
# 🌟 1. 一键自动安装 DeepSeek 官方 Janus 库
# ==========================================
print("📦 正在安装 DeepSeek 官方 Janus 依赖库...")
os.system("pip install -q git+https://github.com/deepseek-ai/Janus.git")
sys.path.append("/opt/conda/lib/python3.10/site-packages") 

import torch
import gc
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

try:
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    from janus.utils.io import load_pil_images
except ImportError:
    print("❌ 官方库加载失败，请确保已经执行过 Restart Kernel！")
    sys.exit()

# ==========================================
# 🧹 2. 显存大扫除
# ==========================================
print("🧹 正在执行显存大扫除...")
for name in list(globals().keys()):
    if name in ['vl_gpt', 'vl_chat_processor', 'inputs_embeds', 'outputs']:
        del globals()[name]
        
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# ==========================================
# 🔧 3. 终极防崩溃补丁区 (修正版)
# ==========================================
transformers.PreTrainedModel.all_tied_weights_keys = {}

original_linspace = torch.linspace
def patched_linspace(*args, **kwargs):
    kwargs["device"] = "cpu" 
    return original_linspace(*args, **kwargs)
torch.linspace = patched_linspace

# ==========================================
# 📊 4. 数据集定义
# ==========================================
class DeepSeekVLMDataset(Dataset):
    def __init__(self, json_path, img_dir):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item['image'])
        raw_label = str(item['label']).lower()
        label = 1 if raw_label in ["1", "real", "true"] else 0
        return img_path, label

def main():
    print("🚀 正在加载 DeepSeek Janus-Pro-7B (官方视觉大模型, 单卡全速版)...")
    
    model_path = "deepseek-ai/Janus-Pro-7B" 
    
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    try:
        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map={"": 0}  # 🌟 核心修复：强行将所有模型权重锁死在 cuda:0，避免跨卡通信报错！
        ).eval()
    finally:
        torch.linspace = original_linspace
        print("✅ 补丁移除，模型成功存活并加载完毕！")

    # ==========================================
    # 🏃 5. 开始极速推理
    # ==========================================
    dataset = DeepSeekVLMDataset(json_path="/kaggle/working/test.json", img_dir="/kaggle/working/test")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    correct = 0
    total = 0
    results_to_save = []
    
    question = "Does the image look real or fake? Answer ONLY with 'real' or 'fake'."

    print(f"🎯 DeepSeek 极速考场开启！准备处理 {len(dataset)} 张图片...")

    with torch.no_grad():
        for paths, labels in tqdm(dataloader, desc="DeepSeek 推理中"):
            img_path = paths[0]
            true_label = labels[0].item()

            try:
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{question}",
                        "images": [img_path],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                
                pil_images = load_pil_images(conversation)
                
                # 强制将输入数据放到模型所在的同一张卡上 (cuda:0)
                prepare_inputs = vl_chat_processor(
                    conversations=conversation, images=pil_images, force_batchify=True
                ).to(vl_gpt.device, dtype=torch.float16)
                
                inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

                outputs = vl_gpt.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=5,    
                    do_sample=False,     
                    use_cache=True       
                )
                
                response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip()

                res_low = response.lower()
                pred = 1 if ("real" in res_low or "authentic" in res_low) else (0 if "fake" in res_low or "ai-generated" in res_low else -1)

                if pred == true_label:
                    correct += 1
                total += 1

                results_to_save.append({
                    "image_path": img_path,
                    "true_label": "real" if true_label == 1 else "fake",
                    "pred_label": "real" if pred == 1 else ("fake" if pred == 0 else "unknown"),
                    "model_response": response
                })

            except Exception as e:
                print(f"\n❌ 出错: {img_path}, 错误: {e}")

    output_file = "/kaggle/working/deepseek_janus_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=4, ensure_ascii=False)

    accuracy = correct / total if total > 0 else 0
    print("\n" + "="*50)
    print(f"🎓 DeepSeek Janus-Pro-7B 最终准确率: {accuracy * 100:.2f}%")
    print(f"📁 详细结果已保存至: {output_file}")
    print("="*50)

if __name__ == "__main__":
    main()
