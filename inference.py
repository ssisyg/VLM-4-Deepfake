import os
import sys
import torch
import requests
from io import BytesIO
from PIL import Image
from transformers import AutoTokenizer, AutoConfig

# =========================================================================
# ⚙️ Configuration
# =========================================================================
TEST_IMAGE_URL = "https://th.bing.com/th/id/OIP.Wkc-8uvRovAZfA-abGrIuAHaHa?w=193&h=186&c=7&r=0&o=7&dpr=1.6&pid=1.7&rm=3"
PROMPT_QUERY = "Does the image looks real/fake?" # Must align with training data format
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.2

# =========================================================================
# 🕵️‍♂️ Phase 1: Dynamic Path Resolution
# =========================================================================
print("🔍 [1/4] Scanning for model paths and dependencies...")

# Automatically locate the modified LLaVA source code
real_path = None
for root, dirs, files in os.walk('/kaggle'):
    if 'llava_llama.py' in files and 'language_model' in root:
        real_path = os.path.dirname(os.path.dirname(os.path.dirname(root)))
        break

if real_path:
    sys.path.insert(0, real_path)
    print(f"   ✅ LLaVA core injected: {real_path}")
else:
    raise FileNotFoundError("Could not locate LLaVA source code directory.")

# Automatically locate Base Model and LoRA Weights
base_path = None
lora_path = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'config.json' in files and 'vicuna' in root.lower():
        base_path = root
    if ('adapter_config.json' in files) and ('llava' in root.lower() or 'srm' in root.lower() or 'weights' in root.lower()):
        lora_path = root

if not base_path or not lora_path:
    raise ValueError("Missing Base Model or LoRA path. Please check dataset mounts.")

print(f"   🎯 Base Model: {base_path}")
print(f"   🎯 LoRA Weights: {lora_path}")

from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from peft import PeftModel
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

# =========================================================================
# 🚀 Phase 2: Model Assembly Pipeline
# =========================================================================
print("📄 [2/4] Initializing configuration and loading models...")
lora_cfg = AutoConfig.from_pretrained(lora_path)

# Remove 8-bit quantization config to enforce float16 loading
if hasattr(lora_cfg, 'quantization_config'):
    del lora_cfg.quantization_config

tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False)

# Load Base LLaMA Model
model = LlavaLlamaForCausalLM.from_pretrained(
    base_path,
    config=lora_cfg, 
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

# Apply LoRA Adapters
print("👓 Applying LoRA adapters...")
model = PeftModel.from_pretrained(model, lora_path)

# Inject Vision Projector Weights
print("👁️ Injecting vision projector (mm_projector)...")
proj_file = os.path.join(lora_path, 'mm_projector.bin')
if not os.path.exists(proj_file):
    proj_file = os.path.join(lora_path, 'non_lora_trainables.bin')

if os.path.exists(proj_file):
    proj_weights = torch.load(proj_file, map_location='cpu')
    proj_weights = {(k[11:] if k.startswith('base_model.') else k): v for k, v in proj_weights.items()}
    if any(k.startswith('model.model.') for k in proj_weights):
        proj_weights = {(k[6:] if k.startswith('model.') else k): v for k, v in proj_weights.items()}
    model.load_state_dict(proj_weights, strict=False)
    print("   ✅ Vision projector successfully injected.")
else:
    print("   ⚠️ WARNING: Vision projector weights not found.")

# Initialize CLIP Vision Tower
print("📸 Initializing CLIP Vision Tower...")
vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model(device_map="auto")

image_processor = vision_tower.image_processor

# =========================================================================
# 🎯 Phase 3: Data Preparation & Inference
# =========================================================================
print(f"🌐 [3/4] Fetching and processing test image from URL...")
try:
    response = requests.get(TEST_IMAGE_URL, timeout=10)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert('RGB')
except Exception as e:
    raise RuntimeError(f"Failed to fetch or open image: {e}")

image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)

print("⚙️ [4/4] Generating Deepfake Analysis Report...")
qs = DEFAULT_IMAGE_TOKEN + '\n' + PROMPT_QUERY
conv = conv_templates["vicuna_v1"].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

with torch.inference_mode():
    output_ids = model.generate(
        inputs=input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        do_sample=True,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True
    )

# Filter out the prompt tokens from the generated output
generated_tokens = output_ids[0][input_ids.shape[1]:]

# =========================================================================
# 📊 Phase 4: Output Rendering
# =========================================================================
print("\n" + "="*60)
print("🎓 【Deepfake AI Analysis Report】")
print("-" * 60)

# Decode only the newly generated tokens
final_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

if not final_response:
    print("⚠️ The model returned an empty response.")
else:
    print(final_response)

print("="*60)
