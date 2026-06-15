import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PEFT_PATH = "outputs/checkpoints/sft_lora_v5_eval_style_completion_only"

OUTPUT_DIR = Path("outputs/demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "sft_v5_inference_with_normalizer_demo.jsonl"


def clean_numeric_answer(x: str) -> str:
    x = str(x).replace("$", "").replace(",", "").strip()
    if re.fullmatch(r"[-+]?\d+\.0+", x):
        x = x.split(".")[0]
    return x


def normalize_final_answer(text: str) -> str:
    """
    Extract the final numeric answer from model output and normalize it as:
    #### <answer>
    This version supports comma-formatted numbers such as $145,000.
    """
    text = str(text)

    number = r"[-+]?\$?\d[\d,]*(?:\.\d+)?"

    # Prefer explicit #### answer if the model already produced it.
    m = re.findall(r"####\s*(" + number + r")", text)
    if m:
        return f"#### {clean_numeric_answer(m[-1])}"

    # Prefer boxed final answers.
    m = re.findall(r"\\boxed\{\\?\$?(" + number + r")\}", text)
    if m:
        return f"#### {clean_numeric_answer(m[-1])}"

    # Handle natural-language zero answers.
    zero_patterns = [
        r"does not need any additional",
        r"doesn't need any additional",
        r"does not need to give any additional",
        r"no remaining feed",
        r"0\\s*\\text\{ cups\}",
        r"=\\s*0\\s*\\text\{ cups\}",
    ]

    tail_text = text[-1000:]
    for pat in zero_patterns:
        if re.search(pat, tail_text, flags=re.IGNORECASE):
            return "#### 0"

    # Then search common final-answer phrases.
    patterns = [
        r"final answer is[^\n]*?(" + number + r")",
        r"answer is[^\n]*?(" + number + r")",
        r"therefore[^\n]*?(" + number + r")",
        r"so[^\n]*?(" + number + r")",
        r"total[^\n]*?(" + number + r")",
        r"needs[^\n]*?(" + number + r")",
        r"makes[^\n]*?(" + number + r")",
    ]

    for pat in patterns:
        ms = re.findall(pat, text, flags=re.IGNORECASE)
        if ms:
            return f"#### {clean_numeric_answer(ms[-1])}"

    # Fallback: use the last number in the generated response.
    nums = re.findall(number, text)
    if nums:
        return f"#### {clean_numeric_answer(nums[-1])}"

    return "#### [invalid]"


def extract_gold_answer(answer_text: str) -> str:
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer_text.replace(",", ""))
    if m:
        return m.group(1)
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", answer_text.replace(",", ""))
    return nums[-1] if nums else "[invalid]"


def main():
    print("====== 加载 tokenizer ======")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print("====== 加载 base model ======")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

    print("====== 加载 SFT_v5 LoRA adapter ======")
    model = PeftModel.from_pretrained(model, PEFT_PATH)
    model.eval()

    print("====== 加载 GSM8K test 样本 ======")
    ds = load_dataset("openai/gsm8k", "main", split="test")

    # 固定选几个样本，方便 demo 可复现。
    sample_ids = [0, 1, 2, 3, 4]

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for idx in sample_ids:
            row = ds[idx]
            question = row["question"]
            gold = extract_gold_answer(row["answer"])

            prompt = f"Q: {question}\nA:"
            messages = [{"role": "user", "content": prompt}]

            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)

            normalized = normalize_final_answer(response)
            normalized_answer = normalized.replace("####", "").strip()
            is_correct = normalized_answer == gold

            out = {
                "sample_id": idx,
                "question": question,
                "gold_answer": gold,
                "model_response": response,
                "normalized_output": normalized,
                "normalized_answer": normalized_answer,
                "is_correct": is_correct,
            }

            f.write(json.dumps(out, ensure_ascii=False) + "\n")

            print("=" * 80)
            print("sample_id:", idx)
            print("question:", question)
            print("gold_answer:", gold)
            print("normalized_output:", normalized)
            print("is_correct:", is_correct)
            print("model_response_head:", response[:500])
            print("model_response_tail:", response[-500:])

    print("====== demo 保存完成 ======")
    print("saved_to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
