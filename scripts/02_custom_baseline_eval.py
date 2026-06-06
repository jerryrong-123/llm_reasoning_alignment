import os
import json
import sys
from pathlib import Path

# ====== 关键：把项目根目录加入 Python 导入路径 ======
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.prompt_template import SYSTEM_PROMPT, build_math_prompt
from src.answer_extract import (
    extract_after_final_answer,
    extract_gsm8k_answer,
    exact_match,
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "eval" / "custom_baseline_gsm8k_sample.jsonl"
BAD_CASE_PATH = PROJECT_ROOT / "outputs" / "bad_cases" / "custom_baseline_gsm8k_bad_cases.jsonl"


def build_messages(question: str):
    """
    把一道数学题包装成 Qwen Instruct 模型需要的 chat 格式。
    """
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_math_prompt(question),
        },
    ]


def main():
    # ====== 创建输出目录 ======
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    os.makedirs(BAD_CASE_PATH.parent, exist_ok=True)

    print("====== 加载 tokenizer ======")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    print("====== 加载模型 ======")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    model.eval()

    print("====== 加载 GSM8K 测试样本 ======")
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="test[:3]",
    )

    correct = 0
    total = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout, open(
        BAD_CASE_PATH, "w", encoding="utf-8"
    ) as fbad:

        for sample in tqdm(dataset):
            question = sample["question"]
            gold_answer = extract_gsm8k_answer(sample["answer"])

            messages = build_messages(question)

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]

            response = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )

            pred_answer = extract_after_final_answer(response)
            is_correct = exact_match(pred_answer, gold_answer)

            correct += int(is_correct)
            total += 1

            record = {
                "question": question,
                "gold_answer": gold_answer,
                "model_response": response,
                "pred_answer": pred_answer,
                "is_correct": is_correct,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if not is_correct:
                fbad.write(json.dumps(record, ensure_ascii=False) + "\n")

    acc = correct / total if total > 0 else 0

    print("\n====== 自定义 Baseline 评估完成 ======")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.4f}")
    print(f"结果保存到: {OUTPUT_PATH}")
    print(f"错误样本保存到: {BAD_CASE_PATH}")


if __name__ == "__main__":
    main()