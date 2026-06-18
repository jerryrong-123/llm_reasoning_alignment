import argparse
import json
from pathlib import Path


def build_format_constrained_text(question: str, reasoning: str, final_answer: str) -> str:
    """
    构造带强格式约束的 SFT 文本。

    目标：
    1. 保留 step-by-step reasoning；
    2. 明确要求最终答案必须用 GSM8K / lm-eval 友好的格式；
    3. 让输出最后一行固定为：#### answer
    """
    return (
        "Please solve the following math problem step by step.\n"
        "You must put the final answer on the last line in exactly this format:\n"
        "#### <final_answer>\n\n"
        f"Question: {question}\n\n"
        "Solution:\n"
        f"{reasoning}\n\n"
        f"#### {final_answer}"
    )


def load_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/processed/sft_small_v2.jsonl",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/processed/sft_small_v2_format.jsonl",
    )
    parser.add_argument(
        "--preview_file",
        type=str,
        default="data/samples/sft_small_v2_format_preview.jsonl",
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    preview_path = Path(args.preview_file)

    if not input_path.exists():
        raise FileNotFoundError(
            f"找不到输入文件: {input_path}。请先确认 sft_small_v2 数据已经生成。"
        )

    print("====== 加载 targeted SFT small_v2 数据 ======")
    print(f"input_file: {input_path}")

    rows = load_jsonl(input_path)

    new_rows = []
    skipped = 0
    category_counts = {}

    print("====== 构造 format-constrained SFT small_v2 数据 ======")

    for row in rows:
        question = str(row.get("question", "")).strip()
        reasoning = str(row.get("reasoning", "")).strip()
        final_answer = str(row.get("final_answer", "")).strip()
        category = str(row.get("category", "unknown")).strip() or "unknown"

        if not question or not reasoning or not final_answer:
            skipped += 1
            continue

        new_row = dict(row)
        new_row["format_constraint"] = "final_line_must_be_gsm8k_hash_answer"
        new_row["text"] = build_format_constrained_text(
            question=question,
            reasoning=reasoning,
            final_answer=final_answer,
        )

        new_rows.append(new_row)
        category_counts[category] = category_counts.get(category, 0) + 1

    if not new_rows:
        raise ValueError("没有构造出有效的 format-constrained SFT 样本。")

    print("")
    print("====== 类别统计 ======")
    for category, count in sorted(category_counts.items()):
        print(f"{category}: {count}")

    print(f"跳过样本数: {skipped}")
    print(f"总样本数: {len(new_rows)}")

    print(f"====== 写入 {output_path} ======")
    write_jsonl(output_path, new_rows)

    print(f"====== 写入预览 {preview_path} ======")
    write_jsonl(preview_path, new_rows[:20])

    print("====== SFT small_v2 format 数据构造完成 ======")
    print(f"train_file: {output_path}")
    print(f"preview_file: {preview_path}")


if __name__ == "__main__":
    main()