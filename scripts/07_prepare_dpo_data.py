import json
import os
from datasets import load_dataset


OUTPUT_PATH = "data/processed/dpo_debug.jsonl"
PREVIEW_PATH = "data/samples/dpo_debug_preview.jsonl"


SYSTEM_PROMPT = """You are a helpful assistant specialized in mathematical reasoning.
Solve the problem step by step.
Give an accurate and clear response."""


def save_jsonl(records, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_prompt(instruction: str) -> str:
    """
    构造 Qwen chat template 风格的 DPO prompt。
    chosen / rejected 会作为 assistant 后续输出。
    """
    return f"""<|im_start|>system
{SYSTEM_PROMPT}
<|im_end|>
<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant
"""


def clean_response(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()

    if not text.endswith("<|im_end|>"):
        text = text + "\n<|im_end|>"

    return text


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/samples", exist_ok=True)

    print("====== 加载 DPO 偏好数据 ======")

    ds = load_dataset(
        "argilla/distilabel-math-preference-dpo",
        split="train[:50]",
    )

    print(ds)
    print("字段:", ds.column_names)

    records = []

    for sample in ds:
        instruction = sample["instruction"]
        chosen_response = sample["chosen_response"]
        rejected_response = sample["rejected_response"]

        if not instruction or not chosen_response or not rejected_response:
            continue

        prompt = build_prompt(instruction)
        chosen = clean_response(chosen_response)
        rejected = clean_response(rejected_response)

        record = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_rating": sample.get("chosen_rating", None),
            "rejected_rating": sample.get("rejected_rating", None),
        }

        records.append(record)

    save_jsonl(records, OUTPUT_PATH)
    save_jsonl(records[:5], PREVIEW_PATH)

    print("\n====== DPO 数据构造完成 ======")
    print(f"总样本数: {len(records)}")
    print(f"训练文件: {OUTPUT_PATH}")
    print(f"预览文件: {PREVIEW_PATH}")

    if len(records) > 0:
        print("\n====== 第一条样本预览 ======")
        print(json.dumps(records[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()