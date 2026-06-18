import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset


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
    parser.add_argument("--hard_file", default="data/processed/grpo_v4_hard_prompts_from_sft_v5.jsonl")
    parser.add_argument("--output", default="data/processed/grpo_v5_mixed_replay_from_sft_v5.jsonl")
    parser.add_argument("--preview", default="data/samples/grpo_v5_mixed_replay_from_sft_v5_preview.jsonl")
    parser.add_argument("--target_total", type=int, default=4000)
    parser.add_argument("--hard_repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    hard_path = PROJECT_ROOT / args.hard_file
    output_path = PROJECT_ROOT / args.output
    preview_path = PROJECT_ROOT / args.preview

    if not hard_path.exists():
        raise FileNotFoundError(f"Missing hard file: {hard_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    hard_rows = []
    with hard_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                hard_rows.append(json.loads(line))

    hard_questions = set(x["question"] for x in hard_rows if "question" in x)

    rows = []

    for r in range(args.hard_repeat):
        for x in hard_rows:
            rows.append(
                {
                    "prompt": x["prompt"],
                    "answer": x["answer"],
                    "source": f"gsm8k_train_sft_v5_wrong_replay_{r+1}",
                    "format_instruction": "End with exactly one final answer line: #### <answer>",
                }
            )

    print("hard_rows:", len(hard_rows))
    print("hard_replay_rows:", len(rows))

    ds = load_dataset("openai/gsm8k", "main", split="train")
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    random_added = 0

    for idx in indices:
        if len(rows) >= args.target_total:
            break

        item = ds[idx]
        question = item["question"].strip()

        if question in hard_questions:
            continue

        answer = extract_final_answer(item["answer"])
        if not question or not answer:
            continue

        rows.append(
            {
                "prompt": f"Q: {question}\nA:",
                "answer": answer,
                "source": "gsm8k_train_random_replay",
                "format_instruction": "End with exactly one final answer line: #### <answer>",
            }
        )
        random_added += 1

    rng.shuffle(rows)

    with output_path.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with preview_path.open("w", encoding="utf-8") as f:
        for x in rows[:5]:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print("====== GRPO_v5 mixed replay 数据完成 ======")
    print("total_records:", len(rows))
    print("hard_original:", len(hard_rows))
    print("hard_repeat:", args.hard_repeat)
    print("hard_replay_records:", len(hard_rows) * args.hard_repeat)
    print("random_replay_records:", random_added)
    print("saved_to:", output_path.relative_to(PROJECT_ROOT))
    print("preview_to:", preview_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
