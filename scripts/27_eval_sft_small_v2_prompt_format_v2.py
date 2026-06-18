import csv
import json
import re
from pathlib import Path
from fractions import Fraction

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PEFT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "sft_lora_small_v2"

LIMIT = 20
MAX_NEW_TOKENS = 512

OUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT_DIR / "sft_small_v2_prompt_format_v2_eval.csv"
JSONL_PATH = OUT_DIR / "sft_small_v2_prompt_format_v2_eval.jsonl"
MD_PATH = OUT_DIR / "sft_small_v2_prompt_format_v2_eval.md"


NUMBER_RE = r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?"


def parse_number(text):
    if text is None:
        return None

    s = str(text).strip()
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("%", "")

    match = re.search(NUMBER_RE, s)
    if not match:
        return None

    raw = match.group(0).replace(",", "")

    try:
        if "/" in raw:
            return float(Fraction(raw))
        return float(raw)
    except Exception:
        return None


def numbers_equal(a, b, eps=1e-6):
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) < eps


def extract_gold_answer(answer_text):
    match = re.search(r"####\s*(" + NUMBER_RE + r")", answer_text)
    if match:
        return match.group(1)
    nums = re.findall(NUMBER_RE, answer_text)
    return nums[-1] if nums else None


def extract_final_answer_pred(response):
    patterns = [
        r"(?:^|\n)\s*Final answer\s*:\s*(" + NUMBER_RE + r")",
        r"(?:^|\n)\s*final answer\s*:\s*(" + NUMBER_RE + r")",
    ]

    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, response, flags=re.IGNORECASE))

    return matches[-1] if matches else None


def extract_hash_pred(response):
    matches = re.findall(r"(?:^|\n)\s*####\s*(" + NUMBER_RE + r")", response)
    return matches[-1] if matches else None


def extract_answer_is_pred(response):
    patterns = [
        r"(?:the answer is|answer is)\s*[:\-]?\s*(" + NUMBER_RE + r")",
        r"(?:答案是)\s*[:：]?\s*(" + NUMBER_RE + r")",
    ]

    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, response, flags=re.IGNORECASE))

    return matches[-1] if matches else None


def extract_flexible_pred(response):
    final_pred = extract_final_answer_pred(response)
    if final_pred is not None:
        return final_pred

    hash_pred = extract_hash_pred(response)
    if hash_pred is not None:
        return hash_pred

    answer_is_pred = extract_answer_is_pred(response)
    if answer_is_pred is not None:
        return answer_is_pred

    nums = re.findall(NUMBER_RE, response)
    return nums[-1] if nums else None


def build_prompt(question):
    system_prompt = (
        "You are a helpful math assistant. "
        "Solve the problem step by step. "
        "Keep the reasoning concise."
    )

    user_prompt = (
        f"Problem:\n{question}\n\n"
        "After solving, end your response with exactly one final line:\n"
        "Final answer: <answer>"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_answer(model, tokenizer, question):
    messages = build_prompt(question)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


def write_markdown(summary, rows):
    lines = []

    lines.append("# SFT small_v2 prompt-level format eval v2")
    lines.append("")
    lines.append("## 实验目的")
    lines.append("")
    lines.append(
        "本实验不重新训练模型，只评估当前最佳 checkpoint "
        "`outputs/checkpoints/sft_lora_small_v2`。"
    )
    lines.append("")
    lines.append(
        "相比 v1 的强制 `#### <answer>`，v2 只在 prompt 末尾温和要求输出 "
        "`Final answer: <answer>`，用于测试是否能减少格式约束对推理正确率的干扰。"
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- 如果 `flexible_acc >= 0.55` 且 `format_acc >= 0.30`，说明温和 prompt 有继续优化价值。")
    lines.append("- 如果 `flexible_acc` 仍明显低于原始 `sft_lora_small_v2` 的 `0.6000`，说明 prompt-level format optimization 仍然会伤害推理，应转向 reward-based format optimization。")
    lines.append("")

    lines.append("## Cases")
    lines.append("")
    lines.append("| doc_id | gold | flexible_pred | final_answer_pred | flexible_correct | final_answer_format_correct |")
    lines.append("|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            f"| {row['doc_id']} | {row['gold_answer']} | {row['flexible_pred']} | "
            f"{row['final_answer_pred']} | {row['flexible_correct']} | "
            f"{row['final_answer_format_correct']} |"
        )

    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("====== 加载 GSM8K test 数据 ======")
    dataset = load_dataset("openai/gsm8k", "main", split=f"test[:{LIMIT}]")
    print(dataset)

    print("====== 加载 tokenizer ======")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("====== 加载 base model ======")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    print("====== 加载 LoRA adapter ======")
    model = PeftModel.from_pretrained(base_model, str(PEFT_PATH))
    model.eval()
    model.to("cpu")

    rows = []

    print("====== 开始 prompt-level format eval v2 ======")
    for idx, example in enumerate(dataset):
        question = example["question"]
        gold_raw = extract_gold_answer(example["answer"])
        gold_num = parse_number(gold_raw)

        print(f"[{idx + 1}/{LIMIT}] doc_id={idx}")

        response = generate_answer(model, tokenizer, question)

        flexible_pred = extract_flexible_pred(response)
        final_answer_pred = extract_final_answer_pred(response)
        hash_pred = extract_hash_pred(response)

        flexible_num = parse_number(flexible_pred)
        final_answer_num = parse_number(final_answer_pred)
        hash_num = parse_number(hash_pred)

        flexible_correct = numbers_equal(flexible_num, gold_num)
        final_answer_format_hit = final_answer_pred is not None
        final_answer_format_correct = numbers_equal(final_answer_num, gold_num)
        strict_hash_correct = numbers_equal(hash_num, gold_num)

        row = {
            "doc_id": idx,
            "question": question,
            "gold_answer": gold_raw,
            "response": response,
            "flexible_pred": flexible_pred,
            "final_answer_pred": final_answer_pred,
            "hash_pred": hash_pred,
            "flexible_correct": int(flexible_correct),
            "final_answer_format_hit": int(final_answer_format_hit),
            "final_answer_format_correct": int(final_answer_format_correct),
            "strict_hash_correct": int(strict_hash_correct),
        }
        rows.append(row)

    total = len(rows)
    flexible_correct_count = sum(r["flexible_correct"] for r in rows)
    final_answer_format_hit_count = sum(r["final_answer_format_hit"] for r in rows)
    final_answer_format_correct_count = sum(r["final_answer_format_correct"] for r in rows)
    strict_hash_correct_count = sum(r["strict_hash_correct"] for r in rows)

    summary = {
        "total": total,
        "flexible_correct": flexible_correct_count,
        "flexible_acc": flexible_correct_count / total if total else 0.0,
        "final_answer_format_hit": final_answer_format_hit_count,
        "final_answer_format_hit_rate": final_answer_format_hit_count / total if total else 0.0,
        "final_answer_format_correct": final_answer_format_correct_count,
        "format_acc": final_answer_format_correct_count / total if total else 0.0,
        "strict_hash_correct": strict_hash_correct_count,
        "strict_hash_acc": strict_hash_correct_count / total if total else 0.0,
    }

    print("====== 写入 CSV ======")
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "doc_id",
                "question",
                "gold_answer",
                "response",
                "flexible_pred",
                "final_answer_pred",
                "hash_pred",
                "flexible_correct",
                "final_answer_format_hit",
                "final_answer_format_correct",
                "strict_hash_correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("====== 写入 JSONL ======")
    with JSONL_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("====== 写入 Markdown ======")
    write_markdown(summary, rows)

    print("====== Summary ======")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("====== 输出文件 ======")
    print(CSV_PATH)
    print(JSONL_PATH)
    print(MD_PATH)


if __name__ == "__main__":
    main()