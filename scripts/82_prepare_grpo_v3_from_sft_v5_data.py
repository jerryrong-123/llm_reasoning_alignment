import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def extract_final_answer(answer: str) -> str:
    text = str(answer)
    m = re.search(r"####\s*(.+)", text)
    if m:
        ans = m.group(1).strip()
    else:
        nums = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|[-+]?\d+\s*/\s*\d+", text)
        ans = nums[-1] if nums else ""

    ans = ans.replace(",", "").replace("$", "").strip()
    if ans.endswith(".0"):
        ans = ans[:-2]
    return ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/grpo_v3_from_sft_v5_eval_style.jsonl")
    parser.add_argument("--preview", default="data/samples/grpo_v3_from_sft_v5_eval_style_preview.jsonl")
    parser.add_argument("--max_samples", type=int, default=400)
    args = parser.parse_args()

    output_path = PROJECT_ROOT / args.output
    preview_path = PROJECT_ROOT / args.preview

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    print("====== 加载 GSM8K train ======")
    ds = load_dataset("openai/gsm8k", "main", split="train")

    rows = []
    skipped = 0

    for row in tqdm(ds):
        question = row["question"].strip()
        answer = extract_final_answer(row["answer"])

        if not question or not answer:
            skipped += 1
            continue

        item = {
            "prompt": f"Q: {question}\nA:",
            "answer": answer,
            "source": "gsm8k_train",
            "format_instruction": "End with exactly one final answer line: #### <answer>",
        }
        rows.append(item)

        if len(rows) >= args.max_samples:
            break

    with output_path.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with preview_path.open("w", encoding="utf-8") as f:
        for x in rows[:5]:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print("====== GRPO_v3 数据构造完成 ======")
    print("total_records:", len(rows))
    print("skipped_records:", skipped)
    print("saved_to:", output_path.relative_to(PROJECT_ROOT))
    print("preview_to:", preview_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
