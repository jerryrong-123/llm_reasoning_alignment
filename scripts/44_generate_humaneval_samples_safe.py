import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval"


def load_humaneval_dataset(split: str):
    """
    Load HumanEval from common Hugging Face dataset names.

    This script is safe sample-only:
    - it does not execute generated code;
    - it does not run unit tests;
    - it does not set HF_ALLOW_CODE_EVAL.
    """
    candidates = [
        "openai/openai_humaneval",
        "openai_humaneval",
    ]

    last_error: Optional[Exception] = None

    for name in candidates:
        try:
            print(f"====== 尝试加载 HumanEval 数据集: {name}, split={split} ======")
            dataset = load_dataset(name, split=split)
            print(dataset)
            print("字段:", list(dataset.features.keys()))
            print(f"====== 成功加载: {name} ======")
            return dataset, name
        except Exception as exc:
            print(f"加载失败: {name}")
            print(f"原因: {repr(exc)}")
            last_error = exc

    raise RuntimeError(
        "无法加载 HumanEval 数据集。请检查网络、datasets 版本或数据集名称。"
    ) from last_error


def build_prompt(item: Dict[str, Any]) -> str:
    task_id = item.get("task_id", "")
    prompt = item.get("prompt", "")

    return (
        "You are given a Python programming problem from HumanEval.\n"
        "Complete the function below.\n"
        "Return only valid Python code.\n"
        "Do not include markdown fences.\n"
        "Do not include explanations.\n"
        "Do not include tests or print statements.\n\n"
        f"Task ID: {task_id}\n\n"
        "Function skeleton:\n"
        f"{prompt}\n"
    )


def remove_markdown_fences(text: str) -> str:
    text = text.strip()

    text = re.sub(r"^```(?:python|py)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    text = text.replace("```python", "")
    text = text.replace("```py", "")
    text = text.replace("```", "")

    return text.strip()


def stop_at_unwanted_sections(code: str) -> str:
    stop_patterns = [
        r"\n\s*#\s*Test",
        r"\n\s*#\s*Example",
        r"\n\s*if\s+__name__\s*==",
        r"\n\s*assert\s+",
        r"\n\s*print\s*\(",
        r"\n\s*def\s+check\s*\(",
        r"\n\s*class\s+Test",
    ]

    cut = len(code)
    for pattern in stop_patterns:
        match = re.search(pattern, code, flags=re.IGNORECASE)
        if match:
            cut = min(cut, match.start())

    return code[:cut].strip()


def extract_code(raw_prediction: str, original_prompt: str) -> str:
    """
    Extract code without executing it.

    HumanEval prompts usually already contain the function skeleton.
    If the model only generates a function body, we combine it with the skeleton.
    If the model generates a full function, we keep the generated function.
    """
    cleaned = remove_markdown_fences(raw_prediction)
    cleaned = stop_at_unwanted_sections(cleaned)

    if "def " in cleaned:
        idx = cleaned.find("def ")
        candidate = cleaned[idx:].strip()
        return candidate

    prompt = original_prompt.rstrip()

    if cleaned:
        return (prompt + "\n" + cleaned).strip()

    return prompt.strip()


def make_output_dir(output_name: str) -> Path:
    output_dir = DEFAULT_OUTPUT_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--no-chat-template", action="store_true")
    return parser.parse_args()


def get_torch_dtype(dtype_name: str):
    dtype_name = dtype_name.lower()

    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16

    raise ValueError(f"Unsupported dtype: {dtype_name}")


def main() -> None:
    args = parse_args()

    if args.device != "cpu":
        print(f"当前 device={args.device}。请确认显存足够。")

    print("====== HumanEval safe sample-only generation ======")
    print("注意：本脚本不执行模型生成代码，不运行测试，不计算 pass@1。")
    print("注意：本脚本不设置 HF_ALLOW_CODE_EVAL。")

    dataset, dataset_name = load_humaneval_dataset(args.split)

    limit = min(args.limit, len(dataset))
    rows = []

    output_name = (
        args.output_name
        or f"code_baseline_qwen25_15b_humaneval_limit{limit}_safe_samples"
    )
    output_dir = make_output_dir(output_name)
    output_file = output_dir / "samples_humaneval_safe_generate_only.jsonl"

    print("====== 加载 tokenizer ======")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("====== 加载 model ======")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=get_torch_dtype(args.dtype),
        trust_remote_code=True,
    )
    model.to(args.device)
    model.eval()

    use_chat_template = not args.no_chat_template

    print("====== 开始生成 ======")

    for idx in range(limit):
        item = dict(dataset[idx])
        task_id = item.get("task_id", f"idx_{idx}")
        entry_point = item.get("entry_point", None)
        original_prompt = item.get("prompt", "")

        user_prompt = build_prompt(item)

        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": user_prompt}]
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            input_text = user_prompt

        inputs = tokenizer(input_text, return_tensors="pt").to(args.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        raw_prediction = tokenizer.decode(new_tokens, skip_special_tokens=True)

        extracted_code = extract_code(raw_prediction, original_prompt)

        row = {
            "dataset": dataset_name,
            "split": args.split,
            "index": idx,
            "task_id": task_id,
            "entry_point": entry_point,
            "prompt": original_prompt,
            "prompt_text": user_prompt,
            "raw_prediction": raw_prediction,
            "extracted_code": extracted_code,
            "safe_generate_only": True,
            "executed": False,
            "passed": None,
            "test": item.get("test", None),
            "canonical_solution": item.get("canonical_solution", None),
        }

        rows.append(row)
        save_jsonl(output_file, rows)

        print(f"[{idx + 1}/{limit}] task_id={task_id} entry_point={entry_point} 已生成")

    print("====== 生成完成 ======")
    print(f"输出文件: {output_file}")
    print("executed=false")
    print("safe_generate_only=true")


if __name__ == "__main__":
    main()