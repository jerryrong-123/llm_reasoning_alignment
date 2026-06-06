import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def extract_gsm8k_answer(answer_text: str) -> str:
    """
    GSM8K 的答案通常长这样：
    reasoning... #### 42

    这里提取 #### 后面的最终答案。
    """
    if answer_text is None:
        return ""

    text = str(answer_text).strip()

    if "####" in text:
        return text.split("####")[-1].strip()

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]

    return text


def build_prompt(question: str) -> str:
    return (
        "Please solve the following math problem step by step. "
        "Put the final answer at the end.\n\n"
        f"Question: {question}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    parser.add_argument("--subset", type=str, default="main")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/processed/grpo_small.jsonl",
    )
    parser.add_argument(
        "--preview_file",
        type=str,
        default="data/samples/grpo_small_preview.jsonl",
    )

    args = parser.parse_args()

    output_path = Path(args.output_file)
    preview_path = Path(args.preview_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    print("====== 加载 GRPO small 数据集 ======")
    print(f"dataset_name: {args.dataset_name}")
    print(f"subset: {args.subset}")
    print(f"split: {args.split}")

    ds = load_dataset(args.dataset_name, args.subset, split=args.split)

    print(ds)
    print("字段:", ds.column_names)

    rows = []
    skipped = 0

    print("====== 构造 GRPO/RLVR small 数据 ======")

    for ex in tqdm(ds, total=min(len(ds), args.max_samples)):
        question = str(ex.get("question", "")).strip()
        answer_raw = str(ex.get("answer", "")).strip()
        final_answer = extract_gsm8k_answer(answer_raw)

        if not question or not final_answer:
            skipped += 1
            continue

        row = {
            "prompt": build_prompt(question),
            "answer": final_answer,
            "source": "gsm8k",
        }

        rows.append(row)

        if len(rows) >= args.max_samples:
            break

    print(f"有效样本数: {len(rows)}")
    print(f"跳过样本数: {skipped}")

    if len(rows) == 0:
        raise ValueError("没有构造出有效 GRPO small 样本，请检查 GSM8K 字段。")

    print(f"====== 写入 {output_path} ======")
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"====== 写入预览 {preview_path} ======")
    with preview_path.open("w", encoding="utf-8") as f:
        for row in rows[:10]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("====== GRPO small 数据构造完成 ======")
    print(f"train_file: {output_path}")
    print(f"preview_file: {preview_path}")


if __name__ == "__main__":
    main()