import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import os

from datasets import load_dataset

from src.answer_extract import extract_gsm8k_answer


OUTPUT_PATH = "data/processed/grpo_debug.jsonl"
PREVIEW_PATH = "data/samples/grpo_debug_preview.jsonl"


def save_jsonl(records, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_prompt(question: str) -> str:
    return f"""You are a helpful assistant specialized in mathematical reasoning.

Solve the following problem step by step.
Put the final answer after 'Final Answer:'.

Problem:
{question}
"""


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/samples", exist_ok=True)

    print("====== 加载 GSM8K，构造 GRPO/RLVR 数据 ======")

    ds = load_dataset(
        "openai/gsm8k",
        "main",
        split="train[:20]",
    )

    records = []

    for sample in ds:
        question = sample["question"]
        gold_answer = extract_gsm8k_answer(sample["answer"])

        if not question or not gold_answer:
            continue

        records.append(
            {
                "prompt": build_prompt(question),
                "answer": gold_answer,
                "source": "gsm8k",
            }
        )

    save_jsonl(records, OUTPUT_PATH)
    save_jsonl(records[:5], PREVIEW_PATH)

    print("\n====== GRPO/RLVR 数据构造完成 ======")
    print(f"总样本数: {len(records)}")
    print(f"训练文件: {OUTPUT_PATH}")
    print(f"预览文件: {PREVIEW_PATH}")

    if len(records) > 0:
        print("\n====== 第一条样本预览 ======")
        print(json.dumps(records[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()