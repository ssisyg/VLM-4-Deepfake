"""
Deepfake Detection VLM Evaluation Script
"""

import os
import sys
import json
import torch
import logging
import argparse
import collections
import transformers
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Set up professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeepfakeEvaluator:
    """
    Base Evaluator Class. 
    Handles data loading, saving, and the general evaluation loop.
    """
    def __init__(self, args):
        self.args = args
        self.model = None
        self.processor = None
        self.tokenizer = None

    def load_data(self):
        """Robust JSON loading to bypass potential formatting issues."""
        if not os.path.exists(self.args.data_path):
            logger.error(f"Data file not found: {self.args.data_path}")
            raise FileNotFoundError(self.args.data_path)

        with open(self.args.data_path, 'r', encoding='utf-8') as f:
            # Using json.loads(f.read()) to bypass any hooked json.load
            full_data = json.loads(f.read())

        if not isinstance(full_data, list):
            logger.error("Parsed JSON is not a list. Please check the data format.")
            sys.exit(1)

        # Slice data if limit is set
        if self.args.max_samples > 0:
            full_data = full_data[:self.args.max_samples]
        
        logger.info(f"Successfully loaded {len(full_data)} samples.")
        return full_data

    def build_prompt(self, img_path):
        """Override this method in subclasses for model-specific prompts."""
        raise NotImplementedError

    def process_and_generate(self, img_path, prompt):
        """Override this method in subclasses for model-specific inference."""
        raise NotImplementedError

    def evaluate(self):
        """Main evaluation loop."""
        data = self.load_data()
        results = []

        logger.info("Starting evaluation...")
        with torch.no_grad():
            for item in tqdm(data, desc="Inferencing"):
                img_key = item.get("image") or item.get("img") or item.get("file_name")
                if not img_key:
                    continue
                
                img_path = os.path.join(self.args.img_dir, img_key)
                if not os.path.exists(img_path):
                    logger.warning(f"Image missing, skipping: {img_path}")
                    continue

                try:
                    response = self.process_and_generate(img_path)
                    results.append({
                        "image_id": img_key,
                        "true_label": item.get("label", "N/A"),
                        "prediction": response
                    })
                except Exception as e:
                    logger.error(f"Failed to process {img_key}: {str(e)}")
                    continue

        self.save_results(results)

    def save_results(self, results):
        os.makedirs(os.path.dirname(self.args.output_path), exist_ok=True)
        with open(self.args.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logger.info(f"Evaluation complete. Results saved to {self.args.output_path}")


class JanusEvaluator(DeepfakeEvaluator):
    """
    Janus-specific implementation.
    Includes necessary runtime patches and specific VLM processing.
    """
    def __init__(self, args):
        super().__init__(args)
        self._apply_patches()
        self._load_model()

    def _apply_patches(self):
        """Inject runtime patches required for Janus & Transformers compatibility."""
        logger.info("Injecting Janus runtime patches...")
        
        # Patch collections for Python 3.10+
        if not hasattr(collections, 'Mapping'):
            import collections.abc
            collections.Mapping = collections.abc.Mapping
            
        # Patch Transformers tied weights
        transformers.PreTrainedModel.all_tied_weights_keys = {}

        # Patch torch.linspace to prevent device mismatch errors
        if not hasattr(torch, '_original_linspace'):
            torch._original_linspace = torch.linspace
            def patched_linspace(*args, **kwargs):
                kwargs["device"] = "cpu"
                return torch._original_linspace(*args, **kwargs)
            torch.linspace = patched_linspace

        # Hook json.load for internal model config loading
        import json as json_mod
        if not hasattr(json_mod, '_original_load'):
            json_mod._original_load = json_mod.load
            def hooked_json_load(fp, *args, **kwargs):
                content = json_mod._original_load(fp, *args, **kwargs)
                return {} if isinstance(content, list) else content
            json_mod.load = hooked_json_load

    def _load_model(self):
        """Load the Janus model and processor."""
        from janus.models import MultiModalityCausalLM, VLChatProcessor
        
        logger.info(f"Loading model: {self.args.model_id} (4-bit: {self.args.use_4bit})")
        
        # Quantization setup
        bnb_config = None
        if self.args.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )

        self.processor = VLChatProcessor.from_pretrained(self.args.model_id)
        self.tokenizer = self.processor.tokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_id,
            trust_remote_code=True,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map={"":"0"} if torch.cuda.is_available() else "cpu"
        ).eval()

    def process_and_generate(self, img_path):
        """Janus-specific forward pass."""
        from janus.utils.io import load_pil_images
        
        # Hardcoded prompt for deepfake detection
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{self.args.prompt}",
                "images": [img_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        pil_imgs = load_pil_images(conversation)
        inputs = self.processor(
            conversations=conversation, images=pil_imgs, force_batchify=True
        ).to(self.model.device, dtype=torch.float16)

        inputs_embeds = self.model.prepare_inputs_embeds(**inputs)

        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.args.max_tokens,
            do_sample=False,
            use_cache=True
        )

        return self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VLMs on Deepfake Detection")
    parser.add_argument("--model_id", type=str, default="deepseek-ai/Janus-Pro-7B", help="HuggingFace model ID")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSON data file")
    parser.add_argument("--img_dir", type=str, required=True, help="Root directory containing images")
    parser.add_argument("--output_path", type=str, default="./results.json", help="Path to save results")
    parser.add_argument("--prompt", type=str, default="Is this image a real photograph or AI-generated? Briefly explain.", help="Text prompt for the VLM")
    parser.add_argument("--max_samples", type=int, default=-1, help="Max samples to evaluate (-1 for all)")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max new tokens to generate")
    parser.add_argument("--use_4bit", action="store_true", help="Enable 4-bit quantization")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # You can easily swap evaluators here based on args.model_id in the future
    evaluator = JanusEvaluator(args)
    evaluator.evaluate()
