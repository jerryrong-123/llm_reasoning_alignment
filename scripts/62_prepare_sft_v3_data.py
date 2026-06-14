import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


CATEGORY_KEYWORDS = {
    "percentage_error": [
        "%",
        "percent",
        "percentage",
        "increase",
        "increased",
        "decrease",
        "decreased",
        "discount",
        "remaining",
        "fraction",
        "half",
        "third",
        "twice",
    ],
    "money_profit_error": [
        "$",
        "dollar",
        "dollars",
        "cost",
        "price",
        "profit",
        "sell",
        "sells",
        "sold",
        "bought",
        "worth",
        "value",
        "market",
        "pay",
        "paid",
        "earn",
        "earned",
    ],
    "unit_rate_error": [
        "per",
        "rate",
        "mph",
        "mile",
        "miles",
        "meter",
        "meters",
        "minute",
        "minutes",
        "hour",
        "hours",
        "gb",
        "cup",
        "cups",
        "each day",
        "per day",
    ],
}


def extract_final_answer(answer_text: str) -> str:
    if answer_text is None:
        return ""

    text = str(answer_text).strip()

    if "####" in text:
        return text.split("####")[-1].strip()

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]

    return text


def clean_reasoning_for_sft(reasoning: str) -> str:
    reasoning = str(reasoning).strip()

    if "####" in reasoning:
        reasoning = reasoning.split("####", 1)[0].strip()

    return reasoning


def build_text(question: str, reasoning: str, final_answer: str) -> str:
    clean_reasoning = clean_reasoning_for_sft(reasoning)
    final_answer = str(final_answer).strip()

    return (
        "Please solve the following math problem step by step. "
        "End your response with exactly one final answer line in the format: #### <answer>.\n\n"
        f"Question: {question.strip()}\n\n"
        "Reasoning:\n"
        f"{clean_reasoning}\n\n"
        f"#### {final_answer}"
    )

def detect_category(question: str) -> str:
    text = str(question).lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return category

    return "general"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    parser.add_argument("--subset", type=str, default="main")
    parser.add_argument("--split", type=str, default="train")

    parser.add_argument("--max_percentage", type=int, default=250)
    parser.add_argument("--max_money_profit", type=int, default=250)
    parser.add_argument("--max_unit_rate", type=int, default=200)
    parser.add_argument("--max_general", type=int, default=300)

    parser.add_argument(
        "--output_file",
        type=str,
        default="data/processed/sft_v3.jsonl",
    )
    parser.add_argument(
        "--preview_file",
        type=str,
        default="data/samples/sft_v3_preview.jsonl",
    )

    args = parser.parse_args()

    output_path = Path(args.output_file)
    preview_path = Path(args.preview_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    category_limits = {
        "percentage_error": args.max_percentage,
        "money_profit_error": args.max_money_profit,
        "unit_rate_error": args.max_unit_rate,
        "general": args.max_general,
    }

    category_counts = {key: 0 for key in category_limits}
    rows = []
    skipped = 0

    print("====== 加载 GSM8K 数据集 ======")
    print(f"dataset_name: {args.dataset_name}")
    print(f"subset: {args.subset}")
    print(f"split: {args.split}")

    ds = load_dataset(args.dataset_name, args.subset, split=args.split)

    print(ds)
    print("字段:", ds.column_names)

    print("====== 构造 targeted SFT v3 数据 ======")

    for ex in tqdm(ds):
        question = str(ex.get("question", "")).strip()
        answer_raw = str(ex.get("answer", "")).strip()

        if not question or not answer_raw:
            skipped += 1
            continue

        category = detect_category(question)

        if category not in category_limits:
            category = "general"

        if category_counts[category] >= category_limits[category]:
            continue

        final_answer = extract_final_answer(answer_raw)

        if not final_answer:
            skipped += 1
            continue

        row = {
            "source": "gsm8k",
            "category": category,
            "question": question,
            "reasoning": answer_raw,
            "final_answer": final_answer,
            "text": build_text(question, answer_raw, final_answer),
        }

        rows.append(row)
        category_counts[category] += 1

        if all(category_counts[k] >= category_limits[k] for k in category_limits):
            break

    if not rows:
        raise ValueError("没有构造出有效的 sft_v3 样本，请检查数据集字段。")

    print("")
    print("====== 类别统计 ======")
    for category, count in category_counts.items():
        print(f"{category}: {count}")

    print(f"跳过样本数: {skipped}")
    print(f"总样本数: {len(rows)}")

    print(f"====== 写入 {output_path} ======")
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"====== 写入预览 {preview_path} ======")
    with preview_path.open("w", encoding="utf-8") as f:
        for row in rows[:20]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("====== SFT v3 targeted 数据构造完成 ======")
    print(f"train_file: {output_path}")
    print(f"preview_file: {preview_path}")


if __name__ == "__main__":
    main()
