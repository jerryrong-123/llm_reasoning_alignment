import json
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


OUTPUT_PATH = Path("data/processed/sft_v6_format_repair.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def extract_final_answer(answer: str) -> str:
    m = re.search(r"####\s*(.+)", answer)
    if m:
        return m.group(1).strip().replace(",", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", answer.replace(",", ""))
    return nums[-1] if nums else ""


def clean_reasoning(answer: str) -> str:
    # 去掉 GSM8K 里的 <<计算=结果>> 标记，避免模型学到奇怪格式
    reasoning = re.sub(r"<<[^>]+>>", "", answer)
    reasoning = re.sub(r"####\s*.+", "", reasoning).strip()
    reasoning = re.sub(r"\n{3,}", "\n\n", reasoning)
    return reasoning


def main():
    print("====== 加载 GSM8K train ======")
    ds = load_dataset("openai/gsm8k", "main", split="train")

    max_samples = 4000
    written = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in tqdm(ds):
            question = row["question"].strip()
            answer = row["answer"].strip()

            final_answer = extract_final_answer(answer)
            reasoning = clean_reasoning(answer)

            if not question or not final_answer or not reasoning:
                continue

            prompt = f"Q: {question}\nA:"
            completion = (
                f" {reasoning}\n\n"
                f"Therefore, the final answer is {final_answer}.\n"
                f"#### {final_answer}"
            )

            out = {
                "source": "gsm8k_train",
                "question": question,
                "prompt": prompt,
                "completion": completion,
                "final_answer": final_answer,
                "text": prompt + completion,
            }

            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

            if written >= max_samples:
                break

    print("written:", written)
    print("saved_to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
