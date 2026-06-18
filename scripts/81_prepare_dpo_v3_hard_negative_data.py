import argparse
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def normalize_answer(text: str) -> str:
    text = str(text).strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        return ""
    x = nums[-1]
    if x.endswith(".0"):
        x = x[:-2]
    return x


def extract_gold_answer(answer: str) -> str:
    m = re.search(r"####\s*(.+)", str(answer))
    if m:
        return normalize_answer(m.group(1))
    return normalize_answer(answer)


def extract_pred_answer(text: str) -> str:
    text = str(text).strip()
    m = re.search(r"####\s*(.+)", text)
    if m:
        return normalize_answer(m.group(1))
    markers = [
        "final answer is",
        "the answer is",
        "answer is",
        "final answer:",
        "answer:",
    ]
    low = text.lower()
    for marker in markers:
        idx = low.rfind(marker)
        if idx >= 0:
            return normalize_answer(text[idx:])
    return normalize_answer(text)


def clean_gsm8k_reasoning(answer: str) -> str:
    reasoning = str(answer)
    reasoning = re.sub(r"<<[^>]+>>", "", reasoning)
    reasoning = re.sub(r"####\s*.+", "", reasoning).strip()
    reasoning = re.sub(r"\n{3,}", "\n\n", reasoning)
    return reasoning.strip()


def build_prompt(question: str) -> str:
    return f"Q: {question.strip()}\nA:"


def build_chosen(answer: str, gold: str) -> str:
    reasoning = clean_gsm8k_reasoning(answer)
    return f" {reasoning}\n\n#### {gold}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--sft_adapter", default="outputs/checkpoints/sft_lora_v5_eval_style_completion_only")
    parser.add_argument("--output", default="data/processed/dpo_v3_hard_negative.jsonl")
    parser.add_argument("--preview", default="data/samples/dpo_v3_hard_negative_preview.jsonl")
    parser.add_argument("--max_source_samples", type=int, default=300)
    parser.add_argument("--target_records", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    sft_adapter = PROJECT_ROOT / args.sft_adapter
    output_path = PROJECT_ROOT / args.output
    preview_path = PROJECT_ROOT / args.preview

    if not sft_adapter.exists():
        raise FileNotFoundError(f"Missing SFT adapter: {sft_adapter}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    print("====== 加载 GSM8K train ======")
    ds = load_dataset("openai/gsm8k", "main", split="train")

    print("====== 加载 tokenizer ======")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("====== 加载 base model + SFT_v5 adapter ======")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(sft_adapter))
    model.eval()

    records = []
    checked = 0
    correct_count = 0
    wrong_count = 0

    batch = []
    metas = []

    def flush_batch():
        nonlocal correct_count, wrong_count

        if not batch:
            return

        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_ids = out[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for meta, rejected in zip(metas, texts):
            rejected = rejected.strip()
            pred = extract_pred_answer(rejected)
            gold = meta["gold"]

            if pred == gold:
                correct_count += 1
                continue

            wrong_count += 1

            if not rejected:
                continue

            item = {
                "prompt": meta["prompt"],
                "chosen": meta["chosen"],
                "rejected": " " + rejected,
                "source": "gsm8k_train_sft_v5_hard_negative",
                "question": meta["question"],
                "final_answer": gold,
                "pred_answer": pred,
                "chosen_rating": 1,
                "rejected_rating": 0,
            }
            records.append(item)

        batch.clear()
        metas.clear()

    print("====== 生成 hard negatives ======")
    for row in tqdm(ds.select(range(min(args.max_source_samples, len(ds))))):
        question = row["question"].strip()
        answer = row["answer"].strip()
        gold = extract_gold_answer(answer)
        prompt = build_prompt(question)
        chosen = build_chosen(answer, gold)

        batch.append(prompt)
        metas.append(
            {
                "question": question,
                "prompt": prompt,
                "chosen": chosen,
                "gold": gold,
            }
        )
        checked += 1

        if len(batch) >= args.batch_size:
            flush_batch()

        if len(records) >= args.target_records:
            break

    flush_batch()

    with output_path.open("w", encoding="utf-8") as f:
        for x in records:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with preview_path.open("w", encoding="utf-8") as f:
        for x in records[:5]:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print("====== DPO_v3 hard-negative 数据完成 ======")
    print("checked_source_samples:", checked)
    print("model_correct_count:", correct_count)
    print("model_wrong_count:", wrong_count)
    print("written_records:", len(records))
    print("saved_to:", output_path.relative_to(PROJECT_ROOT))
    print("preview_to:", preview_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
