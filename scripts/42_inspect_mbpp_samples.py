import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-file",
        type=str,
        default="outputs/eval/code_baseline_qwen25_15b_mbpp_limit5_safe_samples/samples_mbpp_safe_generate_only.jsonl",
    )
    parser.add_argument("--max-code-lines", type=int, default=8)
    args = parser.parse_args()

    sample_path = PROJECT_ROOT / args.sample_file

    if not sample_path.exists():
        raise FileNotFoundError(f"sample 文件不存在: {sample_path}")

    print("====== MBPP safe sample 简洁检查 ======")
    print(f"sample 文件: {sample_path}")
    print()

    records = []

    with sample_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    print(f"样本数: {len(records)}")
    print()

    for item in records:
        index = item.get("index")
        task_id = item.get("task_id")
        prompt_text = item.get("prompt_text")
        test_list = item.get("test_list") or []
        extracted_code = item.get("extracted_code") or ""
        safe_generate_only = item.get("safe_generate_only")
        executed = item.get("executed")

        code_lines = extracted_code.splitlines()
        preview = "\n".join(code_lines[: args.max_code_lines])

        print("=" * 80)
        print(f"index: {index}")
        print(f"task_id: {task_id}")
        print(f"safe_generate_only: {safe_generate_only}")
        print(f"executed: {executed}")
        print(f"test_count: {len(test_list)}")
        print()
        print("题目:")
        print(prompt_text)
        print()
        print("生成代码预览:")
        print(preview)

        if len(code_lines) > args.max_code_lines:
            print("...")

        print()

    print("=" * 80)
    print("检查完成。注意：本脚本只读取 sample，不执行模型生成代码。")


if __name__ == "__main__":
    main()