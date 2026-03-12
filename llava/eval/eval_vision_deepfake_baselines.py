"""
Multi-Model Vision Baseline Evaluator for Deepfake Detection.

This script sequentially evaluates multiple traditional vision models (CNNs/ViTs)
from Hugging Face on a standard deepfake detection dataset. It features a robust
dynamic label resolution engine to align various model outputs (which may have 
conflicting 0/1 definitions for real/fake) to a standardized schema.
"""

import os
import json
import logging
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Configure standard academic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pre-defined baseline models (Keys are aliases, Values are Hugging Face hub paths)
DEFAULT_MODELS = {
    "ViT_Baseline": "dima806/deepfake_vs_real_image_detection",
    "EfficientNet_Baseline": "prithivMLmods/Deep-Fake-Detector-Model",
    "CNN_Fallback": "Wvolf/ViT_Deepfake_Detection"
}

class VisionBaselineDataset(Dataset):
    """
    Standardized dataset loader for Deepfake binary classification.
    """
    def __init__(self, json_path: str, img_dir: str):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Annotation file not found: {json_path}")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.img_dir = img_dir

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item['image'])
        raw_label = str(item['label']).lower()
        # Standardize labels: 1 for Real, 0 for Fake
        label = 1 if raw_label in ["1", "real", "true"] else 0
        return img_path, label


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multiple pure-vision models on Deepfake detection.")
    parser.add_argument("--data_json", type=str, default="/kaggle/working/test.json", help="Path to test annotation JSON.")
    parser.add_argument("--img_dir", type=str, default="/kaggle/working/test", help="Directory containing test images.")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/", help="Directory to save evaluation results.")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size. Can be large (e.g., 32-64) for small vision models.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initialized evaluation on device: {device}")
    
    # 1. Prepare DataLoader
    try:
        dataset = VisionBaselineDataset(json_path=args.data_json, img_dir=args.img_dir)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        logger.info(f"Dataset loaded successfully. Total samples: {len(dataset)}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Evaluation Loop over all predefined models
    for model_alias, hf_model_path in DEFAULT_MODELS.items():
        logger.info("=" * 60)
        logger.info(f"Starting evaluation for baseline: {model_alias} ({hf_model_path})")
        
        try:
            processor = AutoImageProcessor.from_pretrained(hf_model_path)
            model = AutoModelForImageClassification.from_pretrained(hf_model_path).to(device)
            model.eval()
            
            # Dynamic Label Resolution
            # Crucial for preventing 'Label Flipping' errors across different open-source models
            id2label = model.config.id2label
            id2label_lower = {int(k): str(v).lower() for k, v in id2label.items()}
            logger.info(f"Model internal label mapping resolved: {id2label_lower}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_alias} from Hugging Face. Skipping. Error: {e}")
            continue

        correct = 0
        total = 0
        results_to_save = []

        with torch.no_grad():
            for batch_paths, batch_labels in tqdm(dataloader, desc=f"Evaluating {model_alias}"):
                images = []
                valid_paths = []
                valid_labels = []
                
                # Robust image loading
                for path, label in zip(batch_paths, batch_labels):
                    try:
                        img = Image.open(path).convert("RGB")
                        images.append(img)
                        valid_paths.append(path)
                        valid_labels.append(label.item())
                    except Exception as e:
                        logger.warning(f"Skipping corrupted image {path}: {e}")
                        continue
                
                if not images:
                    continue
                    
                # Inference
                try:
                    inputs = processor(images=images, return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    predicted_class_idx = outputs.logits.argmax(-1).cpu().tolist()
                except Exception as e:
                    logger.error(f"Inference failed on current batch. Error: {e}")
                    continue
                
                # Align predictions
                for i in range(len(valid_paths)):
                    pred_id = predicted_class_idx[i]
                    pred_word = id2label_lower.get(pred_id, "unknown")
                    
                    # Standardize to 'real' or 'fake'
                    if any(keyword in pred_word for keyword in ["real", "original", "true"]):
                        final_pred_str = "real"
                    elif any(keyword in pred_word for keyword in ["fake", "forged", "altered"]):
                        final_pred_str = "fake"
                    else:
                        final_pred_str = "unknown"
                        
                    true_str = "real" if valid_labels[i] == 1 else "fake"
                    
                    if final_pred_str == true_str:
                        correct += 1
                    total += 1
                    
                    results_to_save.append({
                        "image_path": valid_paths[i],
                        "true_label": true_str,
                        "pred_label": final_pred_str,
                        "raw_model_output": pred_word
                    })

        # 3. Save Results
        output_file = os.path.join(args.output_dir, f"{model_alias}_results.json")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_to_save, f, indent=4, ensure_ascii=False)
                
            accuracy = (correct / total) * 100 if total > 0 else 0.0
            logger.info(f"Evaluation completed for {model_alias}.")
            logger.info(f"Final Accuracy: {accuracy:.2f}%")
            logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results for {model_alias}: {e}")

    logger.info("=" * 60)
    logger.info("All baseline evaluations completed successfully.")

if __name__ == "__main__":
    main()
