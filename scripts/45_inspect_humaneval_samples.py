import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "eval"
    / "code_baseline_qwen25_15b_humaneval_limit5_safe_samples"
    / "samples_humaneval_safe_generate_only.jsonl"
)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def preview_text(text: str, max_lines: int = 18) -> str:
    lines = str(text).splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines]) + "\n..."


def main():
    print("====== HumanEval safe sample inspection ======")
    print(f"样本文件: {SAMPLE_PATH}")

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"样本文件不存在: {SAMPLE_PATH}")

    rows = load_jsonl(SAMPLE_PATH)
    print(f"样本数: {len(rows)}")

    for row in rows:
        print("\n" + "=" * 80)
        print(f"index: {row.get('index')}")
        print(f"task_id: {row.get('task_id')}")
        print(f"entry_point: {row.get('entry_point')}")
        print(f"safe_generate_only: {row.get('safe_generate_only')}")
        print(f"executed: {row.get('executed')}")
        print("----- extracted_code preview -----")
        print(preview_text(row.get("extracted_code", "")))

    print("\n====== 检查完成 ======")
    print("注意：本脚本只读取样本，不执行模型生成代码。")


if __name__ == "__main__":
    main()