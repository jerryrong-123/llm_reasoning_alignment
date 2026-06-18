import json
from pathlib import Path


INPUT_PATH = Path("outputs/reports/baseline_vs_sft_v5_limit200_sample_comparison.json")
OUTPUT_PATH = Path("outputs/reports/sft_v5_limit200_case_study.md")


def short_text(x, max_len=900):
    x = str(x).replace("\r", "").strip()
    if len(x) <= max_len:
        return x
    return x[:max_len] + "\n...[truncated]"


def write_case(f, title, rows):
    f.write(f"\n## {title}\n\n")
    if not rows:
        f.write("No cases found.\n\n")
        return

    for i, row in enumerate(rows, 1):
        f.write(f"### Case {i}: doc_id={row.get('doc_id')}\n\n")
        f.write(f"**Question**\n\n{short_text(row.get('question', ''), 600)}\n\n")
        f.write(f"**Gold answer**: `{row.get('target')}`\n\n")
        f.write(f"**Baseline correct**: `{row.get('baseline_correct')}`\n\n")
        f.write(f"**SFT_v5 correct**: `{row.get('sft_v5_correct')}`\n\n")
        f.write("**Baseline extracted answer**\n\n")
        f.write(f"```text\n{short_text(row.get('baseline_filtered'), 500)}\n```\n\n")
        f.write("**SFT_v5 extracted answer**\n\n")
        f.write(f"```text\n{short_text(row.get('sft_v5_filtered'), 500)}\n```\n\n")
        f.write("**Baseline response head**\n\n")
        f.write(f"```text\n{short_text(row.get('baseline_response_head'), 900)}\n```\n\n")
        f.write("**SFT_v5 response head**\n\n")
        f.write(f"```text\n{short_text(row.get('sft_v5_response_head'), 900)}\n```\n\n")
        f.write("---\n\n")


def main():
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    improved = data.get("improved_examples", [])
    regressed = data.get("regressed_examples", [])

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        f.write("# SFT_v5 Limit=200 Case Study\n\n")
        f.write("This report summarizes sample-level changes between the base model and SFT_v5 on GSM8K limit=200.\n\n")

        f.write("## Summary\n\n")
        f.write(f"- total_common: {data.get('total_common')}\n")
        f.write(f"- baseline_correct: {data.get('baseline_correct')}\n")
        f.write(f"- sft_v5_correct: {data.get('sft_v5_correct')}\n")
        f.write(f"- baseline_accuracy: {data.get('baseline_accuracy')}\n")
        f.write(f"- sft_v5_accuracy: {data.get('sft_v5_accuracy')}\n")
        f.write(f"- both_correct: {data.get('both_correct')}\n")
        f.write(f"- both_wrong: {data.get('both_wrong')}\n")
        f.write(f"- sft_v5_improved: {data.get('sft_v5_improved')}\n")
        f.write(f"- sft_v5_regressed: {data.get('sft_v5_regressed')}\n")
        f.write(f"- net_gain: {data.get('net_gain')}\n\n")

        write_case(f, "Improved cases: baseline wrong, SFT_v5 correct", improved[:8])
        write_case(f, "Regressed cases: baseline correct, SFT_v5 wrong", regressed[:8])

    print("saved_to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
