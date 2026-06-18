import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_prompt(question: str) -> str:
    question = str(question).strip()
    return (
        "Please solve the following math problem step by step. "
        "End your response with exactly one final answer line in the format: #### <answer>.\n\n"
        f"Question: {question}"
    )


def extract_chosen_from_text(text: str) -> str:
    text = str(text).strip()

    if "Reasoning:\n" in text:
        chosen = text.split("Reasoning:\n", 1)[1].strip()
    else:
        chosen = text

    if "\nFinal answer:" in chosen:
        chosen = chosen.split("\nFinal answer:", 1)[0].strip()

    return chosen.strip()


def parse_number(value: str) -> float:
    value = str(value).strip().replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    if not match:
        raise ValueError(f"Cannot parse number from answer: {value}")
    return float(match.group(0))


def format_number(x: float) -> str:
    if abs(x - int(x)) < 1e-9:
        return str(int(x))
    return str(round(x, 4))


def make_wrong_answer(answer: str, rng: random.Random) -> str:
    gold = parse_number(answer)

    candidates = [
        gold + 1,
        gold - 1,
        gold * 2,
        gold / 2 if gold != 0 else 1,
        gold + 10,
    ]

    candidates = [x for x in candidates if abs(x - gold) > 1e-9]
    wrong = rng.choice(candidates)
    return format_number(wrong)


def remove_existing_final_marker(chosen: str) -> str:
    if "####" in chosen:
        return chosen.split("####", 1)[0].strip()
    return chosen.strip()


def make_rejected(chosen: str, final_answer: str, rng: random.Random) -> str:
    wrong_answer = make_wrong_answer(final_answer, rng)
    reasoning_part = remove_existing_final_marker(chosen)

    rejected_type = rng.choice(
        [
            "wrong_final_answer",
            "wrong_final_answer_with_extra_text",
            "missing_final_marker",
            "duplicated_final_answer",
        ]
    )

    if rejected_type == "wrong_final_answer":
        return f"{reasoning_part}\n\n#### {wrong_answer}"

    if rejected_type == "wrong_final_answer_with_extra_text":
        return (
            f"{reasoning_part}\n\n"
            f"#### {wrong_answer}\n"
            f"The answer is {wrong_answer}."
        )

    if rejected_type == "missing_final_marker":
        return (
            f"{reasoning_part}\n\n"
            f"Final answer: {final_answer}"
        )

    if rejected_type == "duplicated_final_answer":
        return (
            f"{reasoning_part}\n\n"
            f"#### {final_answer}\n"
            f"The final answer is {final_answer}."
        )

    raise ValueError(f"Unknown rejected_type: {rejected_type}")


def convert_record(row: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    question = str(row.get("question", "")).strip()
    final_answer = str(row.get("final_answer", "")).strip()
    text = str(row.get("text", "")).strip()
    source = str(row.get("source", "gsm8k")).strip()
    category = str(row.get("category", "unknown")).strip()

    if not question:
        raise ValueError("Missing question.")
    if not final_answer:
        raise ValueError("Missing final_answer.")
    if not text:
        raise ValueError("Missing text.")

    prompt = build_prompt(question)
    chosen = extract_chosen_from_text(text)
    rejected = make_rejected(chosen, final_answer, rng)

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source": source,
        "category": category,
        "final_answer": final_answer,
        "chosen_rating": 1,
        "rejected_rating": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/sft_v3.jsonl")
    parser.add_argument("--output", type=str, default="data/processed/dpo_v2.jsonl")
    parser.add_argument("--preview", type=str, default="data/samples/dpo_v2_preview.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview_size", type=int, default=5)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output
    preview_path = PROJECT_ROOT / args.preview

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    skipped = 0
    category_counts: Dict[str, int] = {}

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            raw = json.loads(line)

            try:
                item = convert_record(raw, rng)
            except Exception:
                skipped += 1
                continue

            rows.append(item)
            category = item["category"]
            category_counts[category] = category_counts.get(category, 0) + 1

    if not rows:
        raise ValueError("No DPO v2 records were created.")

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with preview_path.open("w", encoding="utf-8") as f:
        for row in rows[: args.preview_size]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("====== DPO v2 数据构造完成 ======")
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