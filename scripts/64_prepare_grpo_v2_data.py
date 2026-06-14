import argparse
import json
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_prompt(question: str) -> str:
    question = str(question).strip()

    return (
        "Please solve the following math problem step by step. "
        "End your response with exactly one final answer line in the format: #### <answer>.\n\n"
        f"Question: {question}"
    )


def convert_record(row: Dict[str, Any]) -> Dict[str, Any]:
    question = str(row.get("question", "")).strip()
    answer = str(row.get("final_answer", "")).strip()
    source = str(row.get("source", "gsm8k")).strip()
    category = str(row.get("category", "unknown")).strip()

    if not question:
        raise ValueError("Missing question field.")

    if not answer:
        raise ValueError("Missing final_answer field.")

    return {
        "prompt": build_prompt(question),
        "answer": answer,
        "source": source,
        "category": category,
        "format_instruction": "End with exactly one final answer line: #### <answer>",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/sft_v3.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/grpo_v2.jsonl",
    )
    parser.add_argument(
        "--preview",
        type=str,
        default="data/samples/grpo_v2_preview.jsonl",
    )
    parser.add_argument(
        "--preview_size",
        type=int,
        default=5,
    )
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output
    preview_path = PROJECT_ROOT / args.preview

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0
    category_counts = {}

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            raw = json.loads(line)

            try:
                item = convert_record(raw)
            except Exception:
                skipped += 1
                continue

            rows.append(item)
            category = item["category"]
            category_counts[category] = category_counts.get(category, 0) + 1

    if not rows:
        raise ValueError("No valid GRPO v2 records were created.")

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with preview_path.open("w", encoding="utf-8") as f:
        for row in rows[: args.preview_size]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("====== GRPO v2 数据构造完成 ======")
    print(f"input_file: {input_path.relative_to(PROJECT_ROOT)}")
    print(f"train_file: {output_path.relative_to(PROJECT_ROOT)}")
    print(f"preview_file: {preview_path.relative_to(PROJECT_ROOT)}")
    print(f"total_records: {len(rows)}")
    print(f"skipped_records: {skipped}")
    print("category_counts:")
    for key, value in sorted(category_counts.items()):
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()