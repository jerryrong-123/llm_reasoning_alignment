import json
from pathlib import Path


EVAL_DIR = Path("outputs/eval")
REPORT_DIR = Path("outputs/reports")
OUTPUT_FILE = REPORT_DIR / "lmeval_samples_preview.jsonl"

MAX_SAMPLES_PER_FILE = 3


def infer_stage(path: Path) -> str:
    """
    根据 lm-eval samples 文件路径推断实验阶段。

    注意：
    更长、更具体的 stage 必须放在更前面。
    例如 sft_lora_small_v2_format 必须早于 sft_lora_small_v2，
    sft_lora_small_v2 必须早于 sft_lora_small。
    否则会被错误归类。
    """
    text = str(path).replace("\\", "/").lower()

    if "sft_lora_small_v2_format" in text:
        return "sft_lora_small_v2_format"

    if "sft_lora_small_v2" in text:
        return "sft_lora_small_v2"

    if "sft_lora_small" in text:
        return "sft_lora_small"

    if "dpo_lora_small" in text:
        return "dpo_lora_small"

    if "grpo_lora_small" in text:
        return "grpo_lora_small"

    if "sft_lora" in text:
        return "sft_lora"

    if "dpo_lora" in text:
        return "dpo_lora"

    if "grpo_lora" in text:
        return "grpo_lora"

    if "baseline" in text:
        return "baseline"

    return "unknown"


def find_sample_files():
    if not EVAL_DIR.exists():
        raise FileNotFoundError(f"找不到评估目录: {EVAL_DIR}")

    files = sorted(EVAL_DIR.rglob("samples_*.jsonl"))
    return files


def load_preview_rows(sample_files):
    rows = []

    for sample_file in sample_files:
        stage = infer_stage(sample_file)

        with sample_file.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= MAX_SAMPLES_PER_FILE:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                preview_item = {
                    "stage": stage,
                    "source_file": str(sample_file),
                }

                preview_item.update(item)
                rows.append(preview_item)

    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    print("====== 找到 lm-eval 样本文件 ======")

    sample_files = find_sample_files()

    for file in sample_files:
        print(f"- {file}")

    rows = load_preview_rows(sample_files)
    write_jsonl(OUTPUT_FILE, rows)

    print("")
    print("====== lm-eval 样本预览保存完成 ======")
    print(f"预览文件: {OUTPUT_FILE}")
    print(f"样本数: {len(rows)}")

    if rows:
        print("")
        print("====== 第一条样本字段预览 ======")
        print(json.dumps(rows[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()