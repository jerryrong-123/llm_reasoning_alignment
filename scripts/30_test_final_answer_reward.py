import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rewards import compute_final_answer_reward


INPUT_JSONL = PROJECT_ROOT / "outputs" / "reports" / "sft_small_v2_prompt_format_v2_eval.jsonl"

OUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUT_CSV = OUT_DIR / "final_answer_reward_test.csv"
OUT_MD = OUT_DIR / "final_answer_reward_test.md"


def load_rows():
    rows = []
    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(rows):
    fieldnames = [
        "doc_id",
        "gold_answer",
        "prediction",
        "correctness",
        "format_hit",
        "extractable",
        "correctness_reward",
        "format_reward",
        "extractability_reward",
        "total_reward",
    ]

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows):
    total = len(rows)

    correctness_count = sum(r["correctness"] for r in rows)
    format_hit_count = sum(r["format_hit"] for r in rows)
    extractable_count = sum(r["extractable"] for r in rows)
    avg_reward = sum(r["total_reward"] for r in rows) / total if total else 0.0

    lines = []

    lines.append("# Final answer reward test")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This report tests the reward function before using it in GRPO/RLVR.")
    lines.append("")
    lines.append("The reward design follows:")
    lines.append("")
    lines.append("```text")
    lines.append("answer correctness reward > final-answer format reward")
    lines.append("```")
    lines.append("")

    lines.append("## Reward definition")
    lines.append("")
    lines.append("```text")
    lines.append("correctness_reward:")
    lines.append("  +1.0 if extracted numeric answer equals gold answer")
    lines.append("")
    lines.append("format_reward:")
    lines.append("  +0.1 if response contains a final-answer format")
    lines.append("")
    lines.append("extractability_reward:")
    lines.append("  +0.1 if a numeric prediction can be extracted")
    lines.append("")
    lines.append("total_reward = correctness_reward + format_reward + extractability_reward")
    lines.append("```")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| total | {total} |")
    lines.append(f"| correctness_count | {correctness_count} |")
    lines.append(f"| correctness_rate | {correctness_count / total if total else 0.0:.4f} |")
    lines.append(f"| format_hit_count | {format_hit_count} |")
    lines.append(f"| format_hit_rate | {format_hit_count / total if total else 0.0:.4f} |")
    lines.append(f"| extractable_count | {extractable_count} |")
    lines.append(f"| extractable_rate | {extractable_count / total if total else 0.0:.4f} |")
    lines.append(f"| avg_total_reward | {avg_reward:.4f} |")
    lines.append("")

    lines.append("## Case table")
    lines.append("")
    lines.append("| doc_id | gold | pred | correct | format_hit | extractable | total_reward |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            f"| {row['doc_id']} | {row['gold_answer']} | {row['prediction']} | "
            f"{row['correctness']} | {row['format_hit']} | {row['extractable']} | "
            f"{row['total_reward']:.1f} |"
        )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_JSONL}")

    source_rows = load_rows()
    reward_rows = []

    for row in source_rows:
        reward_info = compute_final_answer_reward(
            response=row["response"],
            gold_answer=row["gold_answer"],
        )

        reward_rows.append(
            {
                "doc_id": row["doc_id"],
                **reward_info,
            }
        )

    write_csv(reward_rows)
    write_md(reward_rows)

    print("====== final answer reward test done ======")
    print(f"input: {INPUT_JSONL}")
    print(f"csv:   {OUT_CSV}")
    print(f"md:    {OUT_MD}")

    total = len(reward_rows)
    correctness_count = sum(r["correctness"] for r in reward_rows)
    format_hit_count = sum(r["format_hit"] for r in reward_rows)
    avg_reward = sum(r["total_reward"] for r in reward_rows) / total if total else 0.0

    print("====== summary ======")
    print(f"total: {total}")
    print(f"correctness_count: {correctness_count}")
    print(f"correctness_rate: {correctness_count / total if total else 0.0:.4f}")
    print(f"format_hit_count: {format_hit_count}")
    print(f"format_hit_rate: {format_hit_count / total if total else 0.0:.4f}")
    print(f"avg_total_reward: {avg_reward:.4f}")


if __name__ == "__main__":
    main()