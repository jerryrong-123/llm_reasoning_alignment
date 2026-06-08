import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rewards import compute_final_answer_reward, extract_gold_answer


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DPO_ADAPTER_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "dpo_lora_small"
TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "grpo_small.jsonl"

OUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUT_CSV = OUT_DIR / "grpo_format_reward_inspection.csv"
OUT_JSONL = OUT_DIR / "grpo_format_reward_inspection.jsonl"
OUT_MD = OUT_DIR / "grpo_format_reward_inspection.md"

NUM_PROMPTS = 3
NUM_GENERATIONS = 4
MAX_NEW_TOKENS = 256


def normalize_gold_answer(answer_text):
    if answer_text is None:
        return ""

    extracted = extract_gold_answer(str(answer_text))
    if extracted is not None:
        return str(extracted)

    return str(answer_text)


def build_messages(prompt):
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful math assistant. "
                "Solve the problem step by step. "
                "At the end, write exactly one final line in this format: "
                "Final answer: <answer>"
            ),
        },
        {
            "role": "user",
            "content": str(prompt),
        },
    ]


def generate_completions(model, tokenizer, prompt, num_generations):
    messages = build_messages(prompt)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    completions = []

    for _ in range(num_generations):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completions.append(completion.strip())

    return completions


def short_text(text, max_len=240):
    text = str(text).replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def write_reports(rows):
    fieldnames = [
        "prompt_id",
        "generation_id",
        "gold_answer",
        "prediction",
        "correctness",
        "format_hit",
        "extractable",
        "correctness_reward",
        "format_reward",
        "extractability_reward",
        "total_reward",
        "prompt",
        "completion",
    ]

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(rows)
    avg_reward = sum(r["total_reward"] for r in rows) / total if total else 0.0
    reward_values = sorted(set(r["total_reward"] for r in rows))
    correctness_count = sum(r["correctness"] for r in rows)
    format_hit_count = sum(r["format_hit"] for r in rows)
    extractable_count = sum(r["extractable"] for r in rows)

    lines = []

    lines.append("# GRPO format reward inspection")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This report inspects generated completions and reward components before running longer GRPO training.")
    lines.append("")
    lines.append("This v2 inspection uses a stronger final-answer instruction and longer generation length.")
    lines.append("")

    lines.append("## Setup")
    lines.append("")
    lines.append("```text")
    lines.append(f"base_model: {BASE_MODEL}")
    lines.append(f"dpo_adapter_path: {DPO_ADAPTER_PATH.relative_to(PROJECT_ROOT)}")
    lines.append(f"train_file: {TRAIN_FILE.relative_to(PROJECT_ROOT)}")
    lines.append(f"num_prompts: {NUM_PROMPTS}")
    lines.append(f"num_generations: {NUM_GENERATIONS}")
    lines.append(f"max_new_tokens: {MAX_NEW_TOKENS}")
    lines.append("prompt_format_instruction: Final answer: <answer>")
    lines.append("```")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| total_generations | {total} |")
    lines.append(f"| correctness_count | {correctness_count} |")
    lines.append(f"| correctness_rate | {correctness_count / total if total else 0.0:.4f} |")
    lines.append(f"| format_hit_count | {format_hit_count} |")
    lines.append(f"| format_hit_rate | {format_hit_count / total if total else 0.0:.4f} |")
    lines.append(f"| extractable_count | {extractable_count} |")
    lines.append(f"| extractable_rate | {extractable_count / total if total else 0.0:.4f} |")
    lines.append(f"| avg_total_reward | {avg_reward:.4f} |")
    lines.append(f"| unique_reward_values | {reward_values} |")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    if len(reward_values) <= 1:
        lines.append("All inspected completions still received the same reward. GRPO still has no useful advantage signal under this setup.")
    else:
        lines.append("The inspected completions received different rewards. This means the reward function can produce useful variance under sampling.")
    lines.append("")
    lines.append("If format_hit remains 0, the model is not following the explicit final-answer format even with a stronger prompt.")
    lines.append("If correctness remains 0, the main bottleneck is reasoning/answer correctness rather than formatting.")
    lines.append("")

    lines.append("## Cases")
    lines.append("")
    lines.append("| prompt_id | gen_id | gold | pred | correct | format_hit | extractable | total_reward | completion_short |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for row in rows:
        lines.append(
            f"| {row['prompt_id']} | {row['generation_id']} | {row['gold_answer']} | "
            f"{row['prediction']} | {row['correctness']} | {row['format_hit']} | "
            f"{row['extractable']} | {row['total_reward']:.1f} | {short_text(row['completion'])} |"
        )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if not DPO_ADAPTER_PATH.exists():
        raise FileNotFoundError(f"Missing DPO adapter: {DPO_ADAPTER_PATH}")

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Missing train file: {TRAIN_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("====== 加载 GRPO small 数据 ======")
    dataset = load_dataset(
        "json",
        data_files=str(TRAIN_FILE),
        split=f"train[:{NUM_PROMPTS}]",
    )

    print(dataset)
    print("字段:", dataset.column_names)

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

    base_model.config.use_cache = False

    print("====== 加载 DPO LoRA adapter ======")
    model = PeftModel.from_pretrained(
        base_model,
        str(DPO_ADAPTER_PATH),
        is_trainable=False,
    )

    model.eval()
    model.to("cpu")

    rows = []

    print("====== 开始 reward inspection v2 ======")
    for prompt_id, example in enumerate(dataset):
        prompt = example["prompt"]
        gold_answer = normalize_gold_answer(example["answer"])

        print(f"[prompt {prompt_id + 1}/{NUM_PROMPTS}] gold={gold_answer}")

        completions = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            num_generations=NUM_GENERATIONS,
        )

        for generation_id, completion in enumerate(completions):
            reward_info = compute_final_answer_reward(
                response=completion,
                gold_answer=gold_answer,
            )

            row = {
                "prompt_id": prompt_id,
                "generation_id": generation_id,
                "prompt": prompt,
                "completion": completion,
                **reward_info,
            }

            rows.append(row)

            print(
                f"  gen={generation_id} "
                f"pred={row['prediction']} "
                f"correct={row['correctness']} "
                f"format={row['format_hit']} "
                f"extract={row['extractable']} "
                f"reward={row['total_reward']:.1f}"
            )

    write_reports(rows)

    print("====== reward inspection v2 done ======")
    print(f"csv:   {OUT_CSV}")
    print(f"jsonl: {OUT_JSONL}")
    print(f"md:    {OUT_MD}")


if __name__ == "__main__":
    main()