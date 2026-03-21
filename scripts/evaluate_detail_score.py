import os
import re
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 设置标准日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class DeepfakeEvaluator:
    """
    使用 LLM-as-a-Judge 机制评估 VLM 在 Deepfake 检测中的细节解释能力。
    """
    
    def __init__(self, gt_path: str, model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
        """
        初始化评估器，加载 Ground Truth 和裁判大模型。
        
        Args:
            gt_path (str): 标准答案 JSON 文件的路径。
            model_id (str): HuggingFace 上的裁判模型 ID。
        """
        self.gt_text_dict = self._load_ground_truth(gt_path)
        self.model, self.tokenizer = self._initialize_judge_model(model_id)

    def _load_ground_truth(self, gt_path: str) -> Dict[str, str]:
        """安全加载并解析 Ground Truth 文本。"""
        gt_dict = {}
        path = Path(gt_path)
        
        if not path.exists():
            logger.error(f"Ground Truth 文件不存在: {gt_path}")
            raise FileNotFoundError(f"文件未找到: {gt_path}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
                
            for item in gt_data: 
                # 兼容多种键名格式
                img_key = item.get("image") or item.get("img") or item.get("file_name")
                
                gt_text = ""
                if "conversations" in item:
                    for conv in item.get("conversations", []):
                        if conv.get("from") == "gpt":
                            gt_text = conv.get("value", "")
                            break
                            
                if img_key and gt_text:
                    pure_filename = os.path.basename(str(img_key))
                    gt_dict[pure_filename] = gt_text.strip()
                    
            logger.info(f"成功加载了 {len(gt_dict)} 个官方详细解析作为打分基准。")
            return gt_dict
            
        except json.JSONDecodeError as e:
            logger.error(f"解析 Ground Truth JSON 失败: {e}")
            raise

    def _initialize_judge_model(self, model_id: str):
        """初始化带有 4-bit 量化的 LLM 裁判。"""
        logger.info(f"正在加载裁判大模型: {model_id} ...")
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_compute_dtype=torch.float16, 
                bnb_4bit_quant_type="nf4"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                quantization_config=bnb_config, 
                device_map="auto", 
                torch_dtype=torch.float16
            ).eval()
            return model, tokenizer
        except Exception as e:
            logger.error(f"模型加载失败。请检查显存或模型路径: {e}")
            raise

    def _parse_score(self, response: str) -> int:
        """从 LLM 回复中严谨地提取 0-100 的分数。"""
        # 使用更严谨的正则，防止匹配到 "85/100" 中的 100，优先匹配前面的数字
        matches = re.findall(r'\b([0-9]|[1-9][0-9]|100)\b', response)
        if matches:
            try:
                score = int(matches[0])
                return max(0, min(100, score))
            except ValueError:
                return 0
        return 0

    @torch.no_grad()
    def get_score(self, gt_text: str, vlm_analysis: str) -> int:
        """调用 LLM 裁判对单条样本进行打分。"""
        prompt = f"""You are a strict and objective judge evaluating an AI model's image analysis.
Your task is to compare the model's analysis with the Ground Truth and give a score from 0 to 100.

Scoring Criteria:
- 100: Captures almost all specific artifacts and details mentioned in the ground truth.
- 70-90: Captures the main point and several details.
- 30-60: Somewhat correct but vague, missing key specific details.
- 0-20: Completely wrong, irrelevant, or totally misses the point.

Ground Truth Analysis: "{gt_text}"
Model's Analysis: "{vlm_analysis}"

Based on the criteria, output ONLY a single integer between 0 and 100. Do not explain."""

        messages = [
            {"role": "system", "content": "You output only a single integer number."}, 
            {"role": "user", "content": prompt}
        ]
        
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            generated_ids = self.model.generate(
                **model_inputs, 
                max_new_tokens=10, 
                temperature=0.01, 
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            return self._parse_score(response)
            
        except Exception as e:
            logger.warning(f"大模型推理出错，默认返回 0 分: {e}")
            return 0

    def evaluate_directory(self, input_dir: str, output_dir: str):
        """遍历目录并评测所有 VLM 预测文件。"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        target_files = [f for f in input_path.rglob("*.json") if f.name.startswith("answers_")]
        if not target_files:
            logger.warning(f"在 {input_dir} 中未找到符合前缀 'answers_' 的待测 JSON 文件。")
            return

        summary_report: Dict[str, str] = {}

        for file_path in target_files:
            logger.info(f"🎬 正在评测文件: {file_path.name}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    vlm_data = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"文件格式损坏，跳过: {file_path.name}")
                continue

            total_score = 0
            valid_samples = 0
            scored_data: List[Dict[str, Any]] = []

            for item in tqdm(vlm_data, desc=f"Scoring {file_path.name}"):
                img_key = item.get("image_id") or item.get("image") or item.get("id") or item.get("question_id")
                if not img_key: 
                    continue
                    
                pure_filename = os.path.basename(str(img_key))
                gt_text = self.gt_text_dict.get(pure_filename)
                
                if not gt_text:
                    continue
                    
                # 动态提取预测文本
                analysis = None
                for key, value in item.items():
                    if any(k in key.lower() for k in ["response", "analysis", "prediction", "res", "output", "text"]):
                        if isinstance(value, str) and value.strip():
                            analysis = value.strip()
                            break 
                            
                if not analysis:
                    continue
                    
                score = self.get_score(gt_text, analysis)
                total_score += score
                valid_samples += 1

                scored_data.append({
                    "image_id": pure_filename,
                    "ground_truth_text": gt_text,
                    "vlm_analysis": analysis,
                    "qwen_score": score
                })

            if valid_samples > 0:
                avg_score = total_score / valid_samples
                summary_report[file_path.name] = f"{avg_score:.2f} / 100"
            else:
                summary_report[file_path.name] = "0.00 / 100 (数据匹配失败)"
                
            # 保存详细结果
            save_path = output_path / f"scored_{file_path.name}"
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(scored_data, f, indent=4, ensure_ascii=False)

        # 打印排行榜
        self._print_leaderboard(summary_report)

    def _print_leaderboard(self, report: Dict[str, str]):
        """在控制台优雅地打印结果排行榜。"""
        print("\n" + "="*60)
        print(f"{'🏆 回答质量与细节覆盖度排行榜 (满分100)':^55}")
        print("="*60)
        for model_file, score in sorted(report.items(), key=lambda x: x[0]):
            print(f"| {model_file:<30} | {score:>20} |")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="VLM Deepfake 细节解释能力评测脚本 (LLM-as-a-Judge)")
    parser.add_argument("--pred_dir", type=str, default="/kaggle/input", help="存放 VLM 预测结果的文件夹路径")
    parser.add_argument("--gt_path", type=str, default="/kaggle/working/test.json", help="Ground Truth 数据集路径")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/final_scored_reports", help="评测报告输出目录")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="裁判模型 ID")
    
    args = parser.parse_args()
    
    try:
        evaluator = DeepfakeEvaluator(gt_path=args.gt_path, model_id=args.model_id)
        evaluator.evaluate_directory(input_dir=args.pred_dir, output_dir=args.output_dir)
        logger.info("🎉 所有的评测流程已成功结束！")
    except KeyboardInterrupt:
        logger.info("🛑 用户手动中断了评测。")
    except Exception as e:
        logger.error(f"❌ 程序发生致命错误: {e}")

if __name__ == "__main__":
    main()
