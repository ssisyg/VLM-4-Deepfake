"""
LLaVA 1.5 Deepfake Detection Baseline Evaluation Script (Accuracy Only)
Description: 
This script evaluates the official LLaVA-1.5-7b model on a specified Deepfake image dataset. 
It calculates the overall Accuracy and saves the detailed model responses to a JSON file 
for further LLM-as-a-Judge analysis.
"""

import argparse
import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor, 
    LlavaForConditionalGeneration, 
    BitsAndBytesConfig
)

class DeepfakeEvalDataset(Dataset):
    """
    Custom Dataset class for loading images and prompts for Deepfake evaluation.
    """
    def __init__(self, json_path, img_dir):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Annotation file not found: {json_path}")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        
        # Standard prompt for FakeClue / Deepfake detection
        self.prompt_template = "USER: <image>\nDoes the image looks real or fake?\nASSISTANT:"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item['image'])
        
        # Normalize labels: 1 for Real, 0 for Fake
        raw_label = str(item['label']).lower()
        label = 1 if raw_label in ["1", "real", "true"] else 0
        
        return self.prompt_template, label, img_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLaVA 1.5 on Deepfake Dataset (Accuracy Only)")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf", help="HuggingFace Model ID")
    parser.add_argument("--data_json", type=str, default="/kaggle/working/test.json", help="Path to the test JSON file")
    parser.add_argument("--img_dir", type=str, default="/kaggle/working/test", help="Directory containing the images")
    parser.add_argument("--output_file", type=str, default="/kaggle/working/llava_baseline_results.json", help="Path to save the evaluation results")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation (default: 1 for less VRAM)")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max new tokens to generate")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"🚀 Loading model: {args.model_id} (4-bit quantization)...")
    
    # Configure 4-bit quantization to save VRAM
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load model and processor
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id, 
        device_map="auto", 
        quantization_config=quantization_config
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_id)
    
    # Load dataset and dataloader
    dataset = DeepfakeEvalDataset(json_path=args.data_json, img_dir=args.img_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 简化后的指标统计：只记录总数和正确数
    correct_predictions = 0
    total_predictions = 0
    results_to_save = []
    
    print(f"🎯 Evaluation started. Total images to process: {len(dataset)}")
    
    with torch.no_grad():
        for prompts, labels, paths in tqdm(dataloader, desc="Evaluating"):
            img_path = paths[0]
            true_label = labels[0].item() # 1 for real, 0 for fake
            
            try:
                # Load and process image
                raw_image = Image.open(img_path).convert("RGB")
                inputs = processor(
                    text=prompts[0], 
                    images=raw_image, 
                    return_tensors='pt'
                ).to("cuda", torch.float16)
                
                # Generate response
                output = model.generate(**inputs, max_new_tokens=args.max_tokens)
                response = processor.decode(output[0], skip_special_tokens=True)
                
                # Clean prompt from the response
                clean_response = response.split("ASSISTANT:")[-1].strip()
                response_lower = clean_response.lower()
                
                # Parse prediction
                if "real" in response_lower or "authentic" in response_lower:
                    pred = 1
                elif "fake" in response_lower or "ai-generated" in response_lower:
                    pred = 0
                else:
                    pred = -1 # Unknown/Ambiguous
                
                # 统计准确率逻辑
                if pred == true_label:
                    correct_predictions += 1
                total_predictions += 1
                
                # Append to results log
                results_to_save.append({
                    "image_path": img_path,
                    "true_label": "real" if true_label == 1 else "fake",
                    "pred_label": "real" if pred == 1 else ("fake" if pred == 0 else "unknown"),
                    "model_response": clean_response
                })
                
            except Exception as e:
                print(f"\n❌ Error processing image {img_path}: {e}")

    # Save detailed JSON results
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=4, ensure_ascii=False)
    print(f"\n💾 Detailed results saved to: {args.output_file}")

    # Calculate final Accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print("\n" + "="*50)
    print("🎓 LLaVA Baseline Evaluation Results 🎓")
    print(f"✅ Accuracy:  {accuracy * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
