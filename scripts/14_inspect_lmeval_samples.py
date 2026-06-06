import json
from pathlib import Path


EVAL_ROOT = Path("outputs/eval")
REPORT_DIR = Path("outputs/reports")
OUTPUT_FILE = REPORT_DIR / "lmeval_samples_preview.jsonl"

SAMPLES_PER_FILE = 3


STAGE_ORDER = {
    "baseline": 0,
    "sft_lora": 1,
    "dpo_lora": 2,
    "grpo_lora": 3,
    "sft_lora_small": 4,
    "dpo_lora_small": 5,
    "grpo_lora_small": 6,
    "unknown": 99,
}


def infer_stage(path: Path) -> str:
    """
    从 lm-eval samples 文件路径推断实验阶段。

    注意：
    small 阶段必须优先判断。
    否则 sft_lora_small / dpo_lora_small / grpo_lora_small
    会被错误识别成 sft_lora / dpo_lora / grpo_lora。
    """
    text = str(path).replace("\\", "/").lower()

    if "sft_lora_small" in text:
        return "sft_lora_small"
    if "dpo_lora_small" in text:
        return "dpo_lora_small"
    if "grpo_lora_small" in text:
        return "grpo_lora_small"

    if "baseline" in text:
        return "baseline"
    if "sft_lora" in text:
        return "sft_lora"
    if "dpo_lora" in text:
        return "dpo_lora"
    if "grpo_lora" in text:
        return "grpo_lora"

    return "unknown"


def load_jsonl(path: Path, limit: int):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

            if len(rows) >= limit:
                break

    return rows


def compact_sample(sample: dict, source_file: Path) -> dict:
    """
    保留 lm-eval sample 中最有用的字段。

    不同版本 lm-eval 的 samples 字段可能略有不同，
    所以这里不强行假设字段一定存在。
    """
    stage = infer_stage(source_file)

    return {
        "stage": stage,
        "source_file": str(source_file),
        "doc": sample.get("doc"),
        "arguments": sample.get("arguments"),
        "resps": sample.get("resps"),
        "filtered_resps": sample.get("filtered_resps"),
        "target": sample.get("target"),
        "exact_match": sample.get("exact_match"),
        "metrics": sample.get("metrics"),
    }


def collect_samples():
    print("====== 找到 lm-eval 样本文件 ======")

    sample_files = sorted(EVAL_ROOT.rglob("samples_*.jsonl"))

    if not sample_files:
        print("没有找到 samples_*.jsonl。")
        return []

    for path in sample_files:
        print(f"- {path}")

    all_rows = []

    for sample_file in sample_files:
        rows = load_jsonl(sample_file, limit=SAMPLES_PER_FILE)

        for row in rows:
            all_rows.append(compact_sample(row, sample_file))

    all_rows.sort(
        key=lambda row: (
            STAGE_ORDER.get(row["stage"], 99),
            row["source_file"],
        )
    )

    return all_rows


def write_preview(rows):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    rows = collect_samples()

    if not rows:
        return

    write_preview(rows)

    print("")
    print("====== lm-eval 样本预览保存完成 ======")
    print(f"预览文件: {OUTPUT_FILE}")
    print(f"样本数: {len(rows)}")

    print("")
    print("====== 第一条样本字段预览 ======")
    print(json.dumps(rows[0], ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()