import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_JSONL = PROJECT_ROOT / "outputs" / "reports" / "sft_small_v2_prompt_format_v2_eval.jsonl"

OUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUT_CSV = OUT_DIR / "prompt_format_v2_reward_diagnosis.csv"
OUT_MD = OUT_DIR / "prompt_format_v2_reward_diagnosis.md"


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def classify(row):
    flexible_correct = int(row.get("flexible_correct", 0))
    final_answer_format_hit = int(row.get("final_answer_format_hit", 0))
    final_answer_format_correct = int(row.get("final_answer_format_correct", 0))

    if flexible_correct and final_answer_format_correct:
        return "answer_correct_and_format_correct"

    if flexible_correct and not final_answer_format_correct:
        if final_answer_format_hit:
            return "answer_correct_but_final_answer_line_wrong"
        return "answer_correct_but_format_missing"

    if not flexible_correct and final_answer_format_hit:
        return "answer_wrong_but_format_hit"

    return "answer_wrong_and_format_missing"


def short_text(text, max_len=160):
    if text is None:
        return ""
    text = str(text).replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def write_csv(rows):
    import csv

    fieldnames = [
        "doc_id",
        "category",
        "gold_answer",
        "flexible_pred",
        "final_answer_pred",
        "flexible_correct",
        "final_answer_format_hit",
        "final_answer_format_correct",
        "reward_design_note",
    ]

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            category = classify(row)

            if category == "answer_correct_and_format_correct":
                note = "positive example: correctness and format can be rewarded together"
            elif category == "answer_correct_but_format_missing":
                note = "format reward should help, but must not override correctness reward"
            elif category == "answer_correct_but_final_answer_line_wrong":
                note = "parser/format mismatch risk: reward should check extracted final answer carefully"
            elif category == "answer_wrong_but_format_hit":
                note = "danger case: format-only reward may reinforce wrong answers"
            else:
                note = "negative example: both correctness and format need improvement"

            writer.writerow(
                {
                    "doc_id": row["doc_id"],
                    "category": category,
                    "gold_answer": row["gold_answer"],
                    "flexible_pred": row["flexible_pred"],
                    "final_answer_pred": row["final_answer_pred"],
                    "flexible_correct": row["flexible_correct"],
                    "final_answer_format_hit": row["final_answer_format_hit"],
                    "final_answer_format_correct": row["final_answer_format_correct"],
                    "reward_design_note": note,
                }
            )


def write_md(rows):
    total = len(rows)

    categories = {}
    for row in rows:
        category = classify(row)
        categories[category] = categories.get(category, 0) + 1

    lines = []

    lines.append("# Prompt format v2 reward diagnosis")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This report diagnoses the prompt-format-v2 evaluation results before moving to reward-based format optimization."
    )
    lines.append("")
    lines.append("The goal is to decide how to combine answer correctness reward and final-answer format reward.")
    lines.append("")

    lines.append("## Category summary")
    lines.append("")
    lines.append("| category | count | ratio |")
    lines.append("|---|---:|---:|")

    for category, count in sorted(categories.items()):
        ratio = count / total if total else 0.0
        lines.append(f"| {category} | {count} | {ratio:.4f} |")

    lines.append("")
    lines.append("## Key conclusion")
    lines.append("")
    lines.append("The reward design should follow this priority:")
    lines.append("")
    lines.append("```text")
    lines.append("answer correctness reward > final-answer format reward")
    lines.append("```")
    lines.append("")
    lines.append("Reason:")
    lines.append("")
    lines.append("1. Some samples have correct answers but imperfect final-answer formatting.")
    lines.append("2. Some samples may satisfy the final-answer format while still producing wrong answers.")
    lines.append("3. Therefore, a format-only reward is unsafe.")
    lines.append("4. Format reward should be a small auxiliary reward, not the main optimization target.")
    lines.append("")

    lines.append("## Suggested reward structure")
    lines.append("")
    lines.append("```text")
    lines.append("total_reward = correctness_reward + small_format_reward + small_extractability_reward")
    lines.append("```")
    lines.append("")
    lines.append("Suggested weights for the next debug experiment:")
    lines.append("")
    lines.append("```text")
    lines.append("correctness_reward:")
    lines.append("  +1.0 if final numeric answer is correct")
    lines.append("  0.0 otherwise")
    lines.append("")
    lines.append("format_reward:")
    lines.append("  +0.1 if the response contains a final answer line")
    lines.append("  0.0 otherwise")
    lines.append("")
    lines.append("extractability_reward:")
    lines.append("  +0.1 if a numeric answer can be extracted")
    lines.append("  0.0 otherwise")
    lines.append("```")
    lines.append("")
    lines.append("This keeps the maximum auxiliary format reward at 0.2, much smaller than the correctness reward.")
    lines.append("")

    lines.append("## Case table")
    lines.append("")
    lines.append("| doc_id | category | gold | flexible_pred | final_answer_pred |")
    lines.append("|---:|---|---:|---:|---:|")

    for row in rows:
        category = classify(row)
        lines.append(
            f"| {row['doc_id']} | {category} | {row['gold_answer']} | "
            f"{row['flexible_pred']} | {row['final_answer_pred']} |"
        )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_JSONL}")

    rows = load_rows(INPUT_JSONL)

    for row in rows:
        row["category"] = classify(row)

    write_csv(rows)
    write_md(rows)

    print("====== prompt format v2 reward diagnosis done ======")
    print(f"input: {INPUT_JSONL}")
    print(f"csv:   {OUT_CSV}")
    print(f"md:    {OUT_MD}")

    categories = {}
    for row in rows:
        category = row["category"]
        categories[category] = categories.get(category, 0) + 1

    print("====== category summary ======")
    for category, count in sorted(categories.items()):
        print(f"{category}: {count}")


if __name__ == "__main__":
    main()