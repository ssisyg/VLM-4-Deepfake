import torch
import os
import json
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "deepseek-ai/deepseek-vl2-tiny" 

class DeepSeekDataset(Dataset):
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
    print(f"正在加载 DeepSeek-VL2 (4-bit 量化)...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True, 
        quantization_config=quantization_config,
        device_map="auto"
    ).eval()
    
    # DeepSeek-VL2 专用处理器
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    dataset = DeepSeekDataset(json_path="/kaggle/working/test.json", img_dir="/kaggle/working/test")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    correct, total = 0, 0
    results_to_save = []

    print(f"DeepSeek {len(dataset)} 张图片...")

    with torch.no_grad():
        for paths, labels in tqdm(dataloader, desc="DeepSeek 推理中"):
            img_path = paths[0]
            true_label = labels[0].item()

            try:
                # 1. 准备对话格式
                conversation = [
                    {
                        "role": "User",
                        "content": "<image_placeholder>\nDoes this image look real or fake? Answer 'real' or 'fake' first, then explain.",
                        "images": [img_path],
                    },
                    {"role": "Assistant", "content": ""}
                ]

                # 2. 图像预处理 (DeepSeek-VL2 内部会处理缩放，但我们手动转 RGB 确保稳健)
                pil_img = Image.open(img_path).convert("RGB")
                
                # 3. 处理输入
                prepare_inputs = processor(
                    conversations=conversation,
                    images=[pil_img],
                    force_batchify=True,
                    system_prompt=""
                ).to(model.device)

                # 4. 生成回答
                inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
                outputs = model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    max_new_tokens=50,
                    do_sample=False
                )
                
                response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                
                # 5. 判定
                res_low = response.lower()
                pred = 1 if "real" in res_low else (0 if "fake" in res_low else -1)

                if pred == true_label: correct += 1
                total += 1

                results_to_save.append({
                    "image": img_path,
                    "true": "real" if true_label == 1 else "fake",
                    "pred": "real" if pred == 1 else ("fake" if pred == 0 else "unknown"),
                    "response": response
                })

            except Exception as e:
                print(f"Error at {img_path}: {e}")

    # 保存结果
    with open("/kaggle/working/deepseek_results.json", 'w') as f:
        json.dump(results_to_save, f, indent=4)

    print(f"\n 评测完成！DeepSeek 准确率: {(correct/total)*100:.2f}%")

if __name__ == "__main__":
    main()
