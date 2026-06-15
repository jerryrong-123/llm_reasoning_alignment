import argparse
import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def extract_final_answer(answer_text: str) -> str:
    """
    GSM8K answer 通常长这样：
    reasoning...
    #### 72

    这里抽取 #### 后面的最终答案。
    """
    if "####" in answer_text:
        final = answer_text.split("####")[-1].strip()
    else:
        final = answer_text.strip().splitlines()[-1].strip()

    final = final.replace(",", "")
    final = re.sub(r"\s+", " ", final).strip()
    return final


def clean_reasoning(answer_text: str) -> str:
    """
    SFT_v4 不再过度清洗 reasoning。
    只去掉原始的 #### final answer 部分，保留完整推理过程。
    """
    if "####" in answer_text:
        reasoning = answer_text.split("####")[0].strip()
    else:
        reasoning = answer_text.strip()

    reasoning = reasoning.strip()
    return reasoning


def build_text(question: str, reasoning: str, final_answer: str) -> str:
    """
    SFT_v4 目标：
    1. 保留完整 reasoning，避免只学格式；
    2. 强制最后一行统一成 #### <answer>；
    3. 不再额外添加 Final answer: 这种污染格式。
    """
    return (
        "Please solve the following math problem step by step. "
        "Keep the reasoning clear and concise. "
        "End your response with exactly one final answer line in the format: #### <answer>.\n\n"
        f"Question: {question}\n\n"
        f"{reasoning}\n\n"
        f"#### {final_answer}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="openai/gsm8k")
    parser.add_argument("--subset", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_examples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", default="data/processed/sft_v4.jsonl")
    parser.add_argument("--preview_file", default="data/samples/sft_v4_preview.jsonl")
    args = parser.parse_args()

    output_path = PROJECT_ROOT / args.output_file
    preview_path = PROJECT_ROOT / args.preview_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    print("====== 加载 GSM8K 数据集 ======")
    print(f"dataset_name: {args.dataset_name}")
    print(f"subset: {args.subset}")
    print(f"split: {args.split}")

    ds = load_dataset(args.dataset_name, args.subset, split=args.split)

    print(ds)
    print("字段:", ds.column_names)

    indices = list(range(len(ds)))
    random.seed(args.seed)
    random.shuffle(indices)

    selected_indices = indices[: args.max_examples]

    print("====== 构造 SFT v4 数据 ======")
    print(f"max_examples: {args.max_examples}")
    print(f"selected_examples: {len(selected_indices)}")

    records = []
    skipped = 0

    for idx in selected_indices:
        item = ds[idx]
        question = str(item["question"]).strip()
        answer_text = str(item["answer"]).strip()

        final_answer = extract_final_answer(answer_text)
        reasoning = clean_reasoning(answer_text)

        if not question or not reasoning or not final_answer:
            skipped += 1
            continue

        text = build_text(question, reasoning, final_answer)

        records.append(
            {
                "source": "gsm8k",
                "question": question,
                "reasoning": reasoning,
                "final_answer": final_answer,
                "text": text,
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with preview_path.open("w", encoding="utf-8") as f:
        for record in records[:5]:
            f.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")

    print("====== SFT v4 数据构造完成 ======")
    print(f"跳过样本数: {skipped}")
    print(f"总样本数: {len(records)}")
    print(f"train_file: {args.output_file}")
    print(f"preview_file: {args.preview_file}")


if __name__ == "__main__":
    main()