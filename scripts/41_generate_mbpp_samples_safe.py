import argparse
import json
import os
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def strip_code_fence(text: str) -> str:
    """
    去掉 markdown 代码块标记。
    同时处理模型输出被截断导致只有开头 ```python、没有结尾 ``` 的情况。
    """
    text = text.strip()

    text = re.sub(
        r"^\s*```(?:python|py)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def remove_extra_test_code(code: str) -> str:
    """
    尽量删除模型额外输出的测试代码、print 语句和解释性内容。
    当前阶段只保存样本，不执行代码。
    """
    stop_patterns = [
        r"\n\s*#\s*Test cases\b",
        r"\n\s*#\s*Tests\b",
        r"\n\s*print\s*\(",
        r"\n\s*assert\s+",
        r"\n\s*if\s+__name__\s*==",
        r"\n\s*Example\s*:",
        r"\n\s*Explanation\s*:",
    ]

    cut_positions = []

    for pattern in stop_patterns:
        match = re.search(pattern, code, flags=re.IGNORECASE)
        if match:
            cut_positions.append(match.start())

    if cut_positions:
        code = code[: min(cut_positions)]

    return code.strip()


def keep_from_first_function(code: str) -> str:
    """
    如果模型在函数前输出了说明文字，就从第一个 def 开始截取。
    """
    match = re.search(r"\bdef\s+\w+\s*\(", code)

    if match:
        return code[match.start():].strip()

    return code.strip()


def drop_incomplete_last_line(code: str) -> str:
    """
    如果最后一行明显是被截断的字符串或括号，尽量删除最后一行。
    这只是文本清洗，不执行代码。
    """
    lines = code.splitlines()

    if not lines:
        return code

    last = lines[-1].strip()

    suspicious_endings = [
        "(",
        "[",
        "{",
        ",",
        "+",
        "-",
        "*",
        "/",
        "=",
        "==",
        "\"",
        "'",
    ]

    if any(last.endswith(x) for x in suspicious_endings):
        return "\n".join(lines[:-1]).strip()

    if last.count("(") > last.count(")"):
        return "\n".join(lines[:-1]).strip()

    if last.count("[") > last.count("]"):
        return "\n".join(lines[:-1]).strip()

    if last.count("{") > last.count("}"):
        return "\n".join(lines[:-1]).strip()

    return code.strip()


def extract_code(text: str) -> str:
    """
    从模型输出中抽取尽量干净的 Python 函数代码。
    注意：这里只做文本清洗，不执行代码。
    """
    code = strip_code_fence(text)
    code = keep_from_first_function(code)
    code = remove_extra_test_code(code)
    code = drop_incomplete_last_line(code)

    return code.strip()


def get_task_prompt(task: dict) -> str:
    """
    MBPP sanitized 数据集的题目字段是 prompt，不是 text。
    为了兼容不同版本，这里按 prompt -> text -> 空字符串的顺序兜底。
    """
    prompt_text = task.get("prompt")

    if prompt_text is None:
        prompt_text = task.get("text")

    if prompt_text is None:
        prompt_text = ""

    return str(prompt_text).strip()


def build_prompt(task: dict) -> tuple[str, str]:
    """
    构造 MBPP 代码生成 prompt。
    注意：这里只让模型生成代码，不执行代码。
    """
    prompt_text = get_task_prompt(task)
    test_list = task.get("test_list", [])

    tests = "\n".join(str(x) for x in test_list)

    full_prompt = f"""You are a helpful coding assistant.

Write a correct Python function for the following programming task.

Task:
{prompt_text}

The function should satisfy these tests:
{tests}

Requirements:
- Return only Python code.
- Do not include explanations.
- Do not include markdown fences.
- Do not include print statements.
- Do not include test cases.
"""
    return full_prompt, prompt_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "eval"
        / f"code_baseline_qwen25_15b_mbpp_limit{args.limit}_safe_samples"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "samples_mbpp_safe_generate_only.jsonl"

    os.environ["PYTHONUTF8"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    print("====== 安全 MBPP sample-only 生成脚本 ======")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"模型: {args.model}")
    print(f"limit: {args.limit}")
    print(f"device: {args.device}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"输出文件: {output_file}")
    print()
    print("安全说明:")
    print("- 本脚本只生成代码文本")
    print("- 不执行模型生成代码")
    print("- 不运行 MBPP 测试用例")
    print("- 不设置 HF_ALLOW_CODE_EVAL")
    print()

    print("====== 加载 MBPP 数据集 ======")
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    dataset = dataset.select(range(min(args.limit, len(dataset))))

    print(dataset)
    print("字段:", dataset.column_names)

    print("====== 加载 tokenizer ======")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    print("====== 加载 base model ======")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    model.to(args.device)
    model.eval()

    print("====== 开始生成 MBPP samples ======")

    with output_file.open("w", encoding="utf-8") as f:
        for idx, task in enumerate(dataset):
            full_prompt, prompt_text = build_prompt(task)

            messages = [
                {"role": "user", "content": full_prompt},
            ]

            if hasattr(tokenizer, "apply_chat_template"):
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                input_text = full_prompt

            inputs = tokenizer(input_text, return_tensors="pt").to(args.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
            extracted_code = extract_code(prediction)

            record = {
                "index": idx,
                "task_id": task.get("task_id"),
                "source_file": task.get("source_file"),
                "prompt_text": prompt_text,
                "reference_code": task.get("code"),
                "test_imports": task.get("test_imports"),
                "test_list": task.get("test_list"),
                "full_prompt": full_prompt,
                "raw_prediction": prediction,
                "extracted_code": extracted_code,
                "safe_generate_only": True,
                "executed": False,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            print(f"[{idx + 1}/{len(dataset)}] task_id={task.get('task_id')} 已生成")

    print()
    print("====== 生成完成 ======")
    print(f"输出文件: {output_file}")
    print("注意：当前只生成 sample，尚未执行代码测试。")


if __name__ == "__main__":
    main()