import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import os
from typing import Any, Dict, Optional

from datasets import load_dataset

from src.answer_extract import extract_gsm8k_answer, extract_boxed_answer


SYSTEM_PROMPT = """You are a helpful assistant specialized in mathematical reasoning.
Solve the problem step by step.
Put the final answer after 'Final Answer:'."""


OUTPUT_PATH = "data/processed/sft_small.jsonl"
PREVIEW_PATH = "data/samples/sft_small_preview.jsonl"


def save_jsonl(records, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_sft_text(question: str, assistant_response: str) -> str:
    return f"""<|im_start|>system
{SYSTEM_PROMPT}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
{assistant_response}
<|im_end|>
"""


def make_assistant_response(reasoning: str, final_answer: str) -> str:
    reasoning = str(reasoning).strip()
    final_answer = str(final_answer).strip()

    if "Final Answer:" in reasoning:
        return reasoning

    return f"""{reasoning}

Final Answer: {final_answer}"""


def build_record(source: str, question: str, reasoning: str, final_answer: str) -> Optional[Dict[str, Any]]:
    if not question or not reasoning or not final_answer:
        return None

    assistant_response = make_assistant_response(reasoning, final_answer)
    text = build_sft_text(question, assistant_response)

    return {
        "source": source,
        "question": question,
        "reasoning": reasoning,
        "final_answer": final_answer,
        "text": text,
    }


def prepare_gsm8k(max_samples: int = 300):
    print("\n====== 处理 GSM8K small ======")

    ds = load_dataset("openai/gsm8k", "main", split=f"train[:{max_samples}]")

    records = []

    for sample in ds:
        question = sample["question"]
        answer_text = sample["answer"]
        final_answer = extract_gsm8k_answer(answer_text)

        if "####" in answer_text:
            reasoning = answer_text.split("####")[0].strip()
        else:
            reasoning = answer_text.strip()

        record = build_record(
            source="gsm8k",
            question=question,
            reasoning=reasoning,
            final_answer=final_answer,
        )

        if record is not None:
            records.append(record)

    print(f"GSM8K 有效样本数: {len(records)}")
    return records


def prepare_math_algebra(max_samples: int = 300):
    print("\n====== 处理 MATH algebra small ======")

    ds = load_dataset(
        "EleutherAI/hendrycks_math",
        "algebra",
        split=f"train[:{max_samples}]",
    )

    records = []

    for sample in ds:
        question = sample["problem"]
        solution = sample["solution"]
        final_answer = extract_boxed_answer(solution)

        record = build_record(
            source="math_algebra",
            question=question,
            reasoning=solution,
            final_answer=final_answer,
        )

        if record is not None:
            records.append(record)

    print(f"MATH algebra 有效样本数: {len(records)}")
    return records


def get_first_existing(sample: Dict[str, Any], keys):
    for key in keys:
        if key in sample and sample[key] not in [None, ""]:
            value = sample[key]

            if isinstance(value, list):
                if len(value) == 0:
                    continue
                return value[0]

            return value

    return None


def prepare_openr1_math(max_samples: int = 300):
    print("\n====== 处理 OpenR1-Math small ======")

    ds = load_dataset(
        "open-r1/OpenR1-Math-220k",
        split=f"train[:{max_samples}]",
    )

    print("OpenR1 字段:", ds.column_names)

    records = []
    skipped = 0

    for sample in ds:
        question = get_first_existing(
            sample,
            ["question", "problem", "prompt", "instruction"],
        )

        reasoning = get_first_existing(
            sample,
            ["solution", "answer", "response", "completion"],
        )

        if question is None or reasoning is None:
            skipped += 1
            continue

        final_answer = extract_boxed_answer(str(reasoning))

        if not final_answer:
            skipped += 1
            continue

        record = build_record(
            source="openr1_math",
            question=str(question),
            reasoning=str(reasoning),
            final_answer=final_answer,
        )

        if record is not None:
            records.append(record)
        else:
            skipped += 1

    print(f"OpenR1 有效样本数: {len(records)}")
    print(f"OpenR1 跳过样本数: {skipped}")
    return records


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/samples", exist_ok=True)

    all_records = []

    all_records.extend(prepare_gsm8k(max_samples=300))
    all_records.extend(prepare_math_algebra(max_samples=300))

    try:
        all_records.extend(prepare_openr1_math(max_samples=300))
    except Exception as e:
        print("OpenR1 处理失败，先跳过。错误信息:")
        print(repr(e))

    save_jsonl(all_records, OUTPUT_PATH)
    save_jsonl(all_records[:10], PREVIEW_PATH)

    print("\n====== SFT small 数据构造完成 ======")
    print(f"总样本数: {len(all_records)}")
    print(f"训练文件: {OUTPUT_PATH}")
    print(f"预览文件: {PREVIEW_PATH}")

    if len(all_records) > 0:
        print("\n====== 第一条样本预览 ======")
        print(json.dumps(all_records[0], ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()