"""
Evaluate DeepSeek Janus-Pro (7B) on Deepfake Detection Task.

This script performs inference using the Janus-Pro-7B Vision-Language Model
to classify images as real or fake. It includes environment patches for 
running 4-bit quantized models efficiently on single/multi-GPU setups 
(e.g., NVIDIA T4) without Out-Of-Memory (OOM) or cross-device errors.
"""

import os
import sys
import json
import logging
import argparse
from tqdm import tqdm

# Ensure the official Janus library is installed and in path
os.system("pip install -q git+https://github.com/deepseek-ai/Janus.git")
sys.path.append("/opt/conda/lib/python3.10/site-packages")

import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

try:
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    from janus.utils.io import load_pil_images
except ImportError:
    raise ImportError("Failed to load Janus library. Please restart the kernel and try again.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def apply_environment_patches():
    """
    Applies runtime patches to bypass specific bugs in the transformers library
    and PyTorch when using 4-bit quantization and device mappings.
    """
    # Patch 1: Bypass 'all_tied_weights_keys' missing attribute error in 4-bit quantization
    transformers.PreTrainedModel.all_tied_weights_keys = {}

    # Patch 2: Fix device mismatch error in torch.linspace during positional embedding generation
    global original_linspace
    original_linspace = torch.linspace

    def patched_linspace(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_linspace(*args, **kwargs)
    
    torch.linspace = patched_linspace
    logger.info("Environment patches applied successfully.")


def remove_environment_patches():
    """Restores the original PyTorch functions."""
    torch.linspace = original_linspace
    logger.info("Environment patches removed.")


class DeepfakeDetectionDataset(Dataset):
    """
    Custom Dataset for loading images and labels for Deepfake detection.
    """
    def __init__(self, json_path: str, img_dir: str):
        """
        Args:
            json_path (str): Path to the annotation JSON file.
            img_dir (str): Directory containing the images.
        """
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
        # Normalize labels: 1 for Real, 0 for Fake
        label = 1 if raw_label in ["1", "real", "true"] else 0
        return img_path, label


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek Janus-Pro-7B Evaluation for Deepfake Detection")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/Janus-Pro-7B", help="HuggingFace model path")
    parser.add_argument("--data_json", type=str, default="/kaggle/working/test.json", help="Path to test.json")
    parser.add_argument("--img_dir", type=str, default="/kaggle/working/test", help="Directory containing test images")
    parser.add_argument("--output_file", type=str, default="/kaggle/working/deepseek_janus_results.json", help="Output JSON path")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size (recommended: 1 for VLM text generation)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1. Memory Cleanup & Initialization
    logger.info("Initializing environment and clearing GPU memory...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    apply_environment_patches()

    # 2. Load Processor and Model
    logger.info(f"Loading processor and model from: {args.model_path}")
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = vl_chat_processor.tokenizer
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    try:
        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map={"": 0}  # Pin to cuda:0 to avoid cross-device communication overhead
        ).eval()
    finally:
        remove_environment_patches()

    # 3. Prepare Dataloader
    dataset = DeepfakeDetectionDataset(json_path=args.data_json, img_dir=args.img_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f"Dataset loaded. Total samples to evaluate: {len(dataset)}")

    # 4. Evaluation Loop
    correct = 0
    total = 0
    results_to_save = []
    prompt_question = "Does the image look real or fake? Answer ONLY with 'real' or 'fake'."

    logger.info("Starting inference...")
    
    with torch.no_grad():
        for paths, labels in tqdm(dataloader, desc="Evaluating"):
            img_path = paths[0]
            true_label = labels[0].item()

            try:
                # Construct official conversation template
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{prompt_question}",
                        "images": [img_path],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                
                pil_images = load_pil_images(conversation)
                
                # Align data types to float16 to match the quantized model
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

                # Parse prediction
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
                logger.error(f"Error processing image {img_path}: {e}")

    # 5. Save Results & Compute Metrics
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=4, ensure_ascii=False)

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    
    logger.info("-" * 50)
    logger.info("Evaluation Completed!")
    logger.info(f"Total Samples Evaluated: {total}")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info(f"Detailed results saved to: {args.output_file}")
    logger.info("-" * 50)

if __name__ == "__main__":
    main()
