import argparse
import csv
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


FEWSHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.\n#### 6",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.\n#### 39",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.\n#### 8",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.\n#### 9",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29.\n#### 29",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33.\n#### 33",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each costs 5 * 3 = 15 dollars. She has 23 - 15 = 8 dollars left.\n#### 8",
    },
]


def normalize_answer(value):
    if value is None:
        return ""

    text = str(value).strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("％", "%")

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)

    if numbers:
        num = numbers[-1]

        try:
            value_float = float(num)

            if value_float.is_integer():
                return str(int(value_float))

            return str(value_float)
        except Exception:
            return num

    return text.lower().strip()


def extract_gold_answer(answer_text):
    answer_text = str(answer_text)

    if "####" in answer_text:
        return normalize_answer(answer_text.split("####")[-1])

    return normalize_answer(answer_text)


def extract_pred_answer(response):
    response = str(response).strip()

    hash_matches = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", response)

    if hash_matches:
        return normalize_answer(hash_matches[-1])

    return normalize_answer(response)


def strict_hash_correct(response, gold_answer):
    lines = [line.strip() for line in str(response).strip().splitlines() if line.strip()]

    if not lines:
        return False

    last_line = lines[-1]
    match = re.fullmatch(r"####\s*(-?\d+(?:\.\d+)?)", last_line)

    if not match:
        return False

    return normalize_answer(match.group(1)) == normalize_answer(gold_answer)


def build_messages(question):
    messages = [
        {
            "role": "system",
            "content": (
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. "
                "Solve math problems step by step. Always put the final answer on the last line "
                "in exactly this format: #### <answer>"
            ),
        }
    ]

    for ex in FEWSHOT_EXAMPLES:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Solve the following math problem step by step. "
                    "The last line must be exactly: #### <answer>\n\n"
                    f"Q: {ex['question']}\nA:"
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": ex["answer"],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": (
                "Solve the following math problem step by step. "
                "The last line must be exactly: #### <answer>\n\n"
                f"Q: {question}\nA:"
            ),
        }
    )

    return messages


def generate_response(model, tokenizer, messages, max_new_tokens):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if "Q:" in response:
        response = response.split("Q:")[0].strip()

    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0].strip()

    return response.strip()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "doc_id",
        "question",
        "gold_answer",
        "pred_answer",
        "flexible_correct",
        "strict_hash_correct",
        "response",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(path, rows, summary):
    lines = []

    lines.append("# SFT small_v2 Prompt-Level Format Eval")
    lines.append("")
    lines.append(
        "> 本报告不重新训练模型，只在评估 prompt 中加入最终答案格式要求，"
        "测试当前最佳 adapter `outputs/checkpoints/sft_lora_small_v2` 是否能改善 strict 格式输出。"
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| total | {summary['total']} |")
    lines.append(f"| flexible_correct | {summary['flexible_correct']} |")
    lines.append(f"| flexible_acc | {summary['flexible_acc']:.4f} |")
    lines.append(f"| strict_hash_correct | {summary['strict_hash_correct']} |")
    lines.append(f"| strict_hash_acc | {summary['strict_hash_acc']:.4f} |")
    lines.append("")

    lines.append("## Bad Case Preview")
    lines.append("")

    bad_rows = [row for row in rows if not row["flexible_correct"]][:10]

    if not bad_rows:
        lines.append("- None")
        lines.append("")
    else:
        for row in bad_rows:
            lines.append(f"### doc_id {row['doc_id']}")
            lines.append("")
            lines.append(f"- Gold answer: `{row['gold_answer']}`")
            lines.append(f"- Pred answer: `{row['pred_answer']}`")
            lines.append(f"- Flexible correct: `{row['flexible_correct']}`")
            lines.append(f"- Strict hash correct: `{row['strict_hash_correct']}`")
            lines.append("")
            lines.append("Question:")
            lines.append("")
            lines.append(row["question"])
            lines.append("")
            lines.append("Response:")
            lines.append("")
            lines.append("```text")
            lines.append(row["response"])
            lines.append("```")
            lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- 如果 `strict_hash_acc` 明显高于 lm-eval 的 strict-match 0.2000，说明只改 prompt 可能有效。"
    )
    lines.append(
        "- 如果 `flexible_acc` 明显低于 0.6000，说明 prompt 格式约束会干扰推理。"
    )
    lines.append(
        "- 该脚本是 custom prompt-level eval，不等同于 lm-eval 官方 strict-match，只用于判断下一步实验方向。"
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--peft_path", type=str, default="outputs/checkpoints/sft_lora_small_v2")
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    parser.add_argument("--subset", type=str, default="main")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/reports/sft_small_v2_prompt_format_eval.csv",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="outputs/reports/sft_small_v2_prompt_format_eval.jsonl",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default="outputs/reports/sft_small_v2_prompt_format_eval.md",
    )

    args = parser.parse_args()

    print("====== 加载 tokenizer ======")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("====== 加载 base model ======")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        device_map={"": args.device},
        trust_remote_code=True,
    )

    print("====== 加载 LoRA adapter ======")
    model = PeftModel.from_pretrained(base_model, args.peft_path)
    model.eval()

    print("====== 加载 GSM8K test 数据 ======")
    ds = load_dataset(args.dataset_name, args.subset, split=args.split)
    ds = ds.select(range(min(args.limit, len(ds))))

    rows = []

    print("====== 开始 prompt-level format eval ======")

    for idx, item in enumerate(ds):
        question = str(item["question"]).strip()
        gold_answer = extract_gold_answer(item["answer"])
        messages = build_messages(question)

        print(f"[{idx + 1}/{len(ds)}] doc_id={idx}")

        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
        )

        pred_answer = extract_pred_answer(response)
        flexible_correct = pred_answer == gold_answer
        strict_correct = strict_hash_correct(response, gold_answer)

        rows.append(
            {
                "doc_id": idx,
                "question": question,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "flexible_correct": flexible_correct,
                "strict_hash_correct": strict_correct,
                "response": response,
            }
        )

    total = len(rows)
    flexible_correct = sum(1 for row in rows if row["flexible_correct"])
    strict_hash_correct_count = sum(1 for row in rows if row["strict_hash_correct"])

    summary = {
        "total": total,
        "flexible_correct": flexible_correct,
        "flexible_acc": flexible_correct / total if total else 0.0,
        "strict_hash_correct": strict_hash_correct_count,
        "strict_hash_acc": strict_hash_correct_count / total if total else 0.0,
    }

    output_csv = Path(args.output_csv)
    output_jsonl = Path(args.output_jsonl)
    output_md = Path(args.output_md)

    write_csv(output_csv, rows)
    write_jsonl(output_jsonl, rows)
    write_markdown(output_md, rows, summary)

    print("")
    print("====== prompt-level format eval 完成 ======")
    print(f"CSV 文件: {output_csv}")
    print(f"JSONL 文件: {output_jsonl}")
    print(f"Markdown 文件: {output_md}")
    print("")
    print("====== 汇总 ======")
    print(summary)


if __name__ == "__main__":
    main()