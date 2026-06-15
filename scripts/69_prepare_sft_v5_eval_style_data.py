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
    if "####" in answer_text:
        final = answer_text.split("####")[-1].strip()
    else:
        final = answer_text.strip().splitlines()[-1].strip()

    final = final.replace(",", "")
    final = re.sub(r"\s+", " ", final).strip()
    return final


def clean_reasoning(answer_text: str) -> str:
    if "####" in answer_text:
        reasoning = answer_text.split("####")[0].strip()
    else:
        reasoning = answer_text.strip()

    reasoning = re.sub(r"\n{3,}", "\n\n", reasoning)
    return reasoning.strip()


def build_prompt(question: str) -> str:
    return f"Q: {question.strip()}\nA:"


def build_completion(reasoning: str, final_answer: str) -> str:
    return f" {reasoning.strip()}\n#### {final_answer.strip()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="openai/gsm8k")
    parser.add_argument("--subset", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_examples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", default="data/processed/sft_v5_eval_style.jsonl")
    parser.add_argument("--preview_file", default="data/samples/sft_v5_eval_style_preview.jsonl")
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

    print("====== 构造 SFT v5 eval-style completion-only 数据 ======")
    print(f"max_examples: {args.max_examples}")
    print(f"selected_examples: {len(selected_indices)}")

    records = []
    skipped = 0

    for idx in selected_indices:
        item = ds[idx]
        question = str(item["question"]).strip()
        answer_text = str(item["answer"]).strip()

        reasoning = clean_reasoning(answer_text)
        final_answer = extract_final_answer(answer_text)

        if not question or not reasoning or not final_answer:
            skipped += 1
            continue

        prompt = build_prompt(question)
        completion = build_completion(reasoning, final_answer)
        text = prompt + completion

        records.append(
            {
                "source": "gsm8k",
                "question": question,
                "reasoning": reasoning,
                "final_answer": final_answer,
                "prompt": prompt,
                "completion": completion,
                "text": text,
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with preview_path.open("w", encoding="utf-8") as f:
        for record in records[:5]:
            f.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")

    print("====== SFT v5 数据构造完成 ======")
    print(f"跳过样本数: {skipped}")
    print(f"总样本数: {len(records)}")
    print(f"train_file: {args.output_file}")
    print(f"preview_file: {args.preview_file}")


if __name__ == "__main__":
    main()