"""
Multi-Model Vision Baseline Evaluator for LOKI Deepfake Detection Dataset.

This script evaluates multiple traditional vision models (ViT, EfficientNet, CNNs) 
from the Hugging Face Hub against the LOKI dataset's true/false image split. 
It features a robust file-indexing engine, dynamic label resolution, and fault-tolerant inference.

Usage:
    python evaluate_loki_deepfake.py \
        --data_json /path/to/true_or_false.json \
        --img_dir /path/to/loki_images \
        --output_dir ./results \
        --batch_size 32
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Tuple
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Pre-defined baseline models (Alias: Hugging Face Model ID)
DEFAULT_MODELS = {
    "ViT_Baseline": "dima806/deepfake_vs_real_image_detection",
    "EfficientNet_Baseline": "prithivMLmods/Deep-Fake-Detector-Model",
    "CNN_Fallback": "Wvolf/ViT_Deepfake_Detection"
}

# -----------------------------------------------------------------------------
# Dataset Engine
# -----------------------------------------------------------------------------
class LOKIVisionDataset(Dataset):
    """
    Standardized dataset loader specifically adapted for the LOKI dataset JSON schema.
    Includes a global file indexer to prevent path mismatch errors.
    """
    def __init__(self, json_path: str, img_dir: str):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Annotation JSON not found at: {json_path}")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        self.img_dir = img_dir
        self.img_index = self._build_global_file_index(img_dir)

    def _build_global_file_index(self, root_dir: str) -> Dict[str, str]:
        """Builds a hash map of filename -> absolute path to ensure robust file loading."""
        logger.info(f"Building global file index from {root_dir}...")
        index = {}
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    index[file] = os.path.join(root, file)
        logger.info(f"Successfully indexed {len(index)} images.")
        return index

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        item = self.data[idx]
        
        # Determine Ground Truth based on LOKI's "question_type"
        q_type = str(item.get('question_type', '')).lower()
        if "_image_real_" in q_type:
            label = 1  # Real / Pristine
        elif "_image_fake_" in q_type:
            label = 0  # Fake / Generated
        else:
            label = 0  # Fallback assumption
            
        # Robust path resolution using the global index
        rel_path = item.get('image_path', '')
        img_name = os.path.basename(rel_path)
        actual_img_path = self.img_index.get(img_name, os.path.join(self.img_dir, rel_path))
            
        return actual_img_path, label

# -----------------------------------------------------------------------------
# Core Evaluation Logic
# -----------------------------------------------------------------------------
def evaluate_model(
    model_alias: str, 
    hf_model_path: str, 
    dataloader: DataLoader, 
    device: torch.device, 
    output_dir: str
):
    """Runs the evaluation loop for a single model and saves the predictions."""
    logger.info("-" * 60)
    logger.info(f"Starting evaluation: {model_alias} ({hf_model_path})")
    
    try:
        processor = AutoImageProcessor.from_pretrained(hf_model_path)
        model = AutoModelForImageClassification.from_pretrained(hf_model_path).to(device)
        model.eval()
        
        # Resolve model's internal class mappings
        id2label_lower = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        logger.info(f"Model internal label mapping: {id2label_lower}")
    except Exception as e:
        logger.error(f"Failed to load model {model_alias}. Skipping. Error: {e}")
        return

    correct, total_valid = 0, 0
    results_to_save = []
    error_count = 0
    MAX_ERROR_LOGS = 5

    with torch.no_grad():
        for batch_paths, batch_labels in tqdm(dataloader, desc=f"Evaluating {model_alias}", unit="batch"):
            images, valid_paths, valid_labels = [], [], []
            
            # Robust Image Loading
            for path, label in zip(batch_paths, batch_labels):
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                    valid_paths.append(path)
                    valid_labels.append(label.item())
                except Exception as e:
                    if error_count < MAX_ERROR_LOGS:
                        logger.warning(f"Failed to read image [{path}]: {e}")
                        error_count += 1
                    continue
            
            if not images:
                continue
                
            # Model Inference
            try:
                inputs = processor(images=images, return_tensors="pt").to(device)
                outputs = model(**inputs)
                predicted_class_idx = outputs.logits.argmax(-1).cpu().tolist()
            except Exception as e:
                logger.error(f"Inference crash on batch. Skipping. Error: {e}")
                continue
            
            # Prediction Alignment
            for i in range(len(valid_paths)):
                pred_word = id2label_lower.get(predicted_class_idx[i], "unknown")
                
                # Standardize semantics to 'real' or 'fake'
                if any(kw in pred_word for kw in ["real", "original", "true", "human"]):
                    final_pred = "real"
                elif any(kw in pred_word for kw in ["fake", "forged", "altered", "spoof", "ai"]):
                    final_pred = "fake"
                else:
                    final_pred = "unknown"
                    
                true_str = "real" if valid_labels[i] == 1 else "fake"
                
                if final_pred == true_str:
                    correct += 1
                total_valid += 1
                
                results_to_save.append({
                    "image_path": valid_paths[i],
                    "true_label": true_str,
                    "pred_label": final_pred,
                    "raw_model_output": pred_word
                })

    # Metrics & I/O
    if total_valid > 0:
        accuracy = (correct / total_valid) * 100
        logger.info(f"[{model_alias}] Evaluation Complete. Accuracy: {accuracy:.2f}% ({correct}/{total_valid})")
    else:
        logger.warning(f"[{model_alias}] Evaluation Complete. Accuracy: 0.00% (No valid images processed).")

    output_file = os.path.join(output_dir, f"{model_alias}_loki_results.json")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=4, ensure_ascii=False)
        logger.info(f"Predictions saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results to disk: {e}")

# -----------------------------------------------------------------------------
# Main Execution Entrypoint
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Model Vision Evaluator for Deepfake Detection.")
    parser.add_argument("--data_json", type=str, required=True, help="Path to the LOKI true_or_false.json annotation file.")
    parser.add_argument("--img_dir", type=str, required=True, help="Root directory containing the unzipped images.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save the evaluation JSONs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size.")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Engine initialized on device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        dataset = LOKIVisionDataset(json_path=args.data_json, img_dir=args.img_dir)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        logger.info(f"Dataset securely loaded. Total testing samples: {len(dataset)}")
    except Exception as e:
        logger.error(f"Fatal Error during dataset initialization: {e}")
        return

    for model_alias, hf_model_path in DEFAULT_MODELS.items():
        evaluate_model(
            model_alias=model_alias, 
            hf_model_path=hf_model_path, 
            dataloader=dataloader, 
            device=device, 
            output_dir=args.output_dir
        )
        
    logger.info("-" * 60)
    logger.info("All model evaluations completed successfully.")

if __name__ == "__main__":
    main()
