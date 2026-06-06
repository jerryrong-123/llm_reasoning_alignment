import json
from pathlib import Path


EVAL_ROOT = Path("outputs/eval")
REPORT_DIR = Path("outputs/reports")
OUTPUT_PATH = REPORT_DIR / "lmeval_samples_preview.jsonl"


def infer_stage(path: Path) -> str:
    text = str(path).lower()

    if "baseline_qwen" in text:
        return "baseline"
    if "sft_lora" in text:
        return "sft_lora"
    if "dpo_lora" in text:
        return "dpo_lora"
    if "grpo_lora" in text:
        return "grpo_lora"

    return "unknown"


def is_lmeval_sample_file(path: Path) -> bool:
    """
    只保留 lm-eval --log_samples 生成的样本文件。

    lm-eval 样本文件一般长这样：
    samples_gsm8k_cot_xxx.jsonl

    跳过我们自定义评估生成的：
    custom_baseline_gsm8k_sample.jsonl
    """
    name = path.name.lower()
    path_text = str(path).lower()

    if "custom_baseline" in path_text:
        return False

    if not name.startswith("samples_"):
        return False

    if not name.endswith(".jsonl"):
        return False

    return True


def find_lmeval_sample_files():
    sample_files = []

    for path in EVAL_ROOT.rglob("*.jsonl"):
        if is_lmeval_sample_file(path):
            sample_files.append(path)

    return sample_files


def load_jsonl(path: Path, max_lines: int = 3):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break

            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except Exception:
                continue

    return records


def simplify_sample(stage: str, path: Path, sample: dict):
    """
    lm-eval 不同版本样本字段可能不同。
    这里尽量保留最有用的信息。
    """
    doc = sample.get("doc", {})
    arguments = sample.get("arguments", None)
    resps = sample.get("resps", None)
    filtered_resps = sample.get("filtered_resps", None)
    target = sample.get("target", None)

    metrics = {}

    for key, value in sample.items():
        if isinstance(value, (int, float, bool)) and (
            "exact" in key or "acc" in key or "match" in key
        ):
            metrics[key] = value

    return {
        "stage": stage,
        "source_file": str(path),
        "doc": doc,
        "arguments": arguments,
        "resps": resps,
        "filtered_resps": filtered_resps,
        "target": target,
        "metrics": metrics,
        "raw_keys": list(sample.keys()),
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sample_files = find_lmeval_sample_files()

    if not sample_files:
        print("没有找到 lm-eval 样本 jsonl 文件。")
        print("请确认评估时使用了 --log_samples。")
        return

    print("====== 找到 lm-eval 样本文件 ======")
    for path in sample_files:
        print("-", path)

    simplified_records = []

    for path in sample_files:
        stage = infer_stage(path)
        samples = load_jsonl(path, max_lines=3)

        for sample in samples:
            simplified_records.append(
                simplify_sample(stage=stage, path=path, sample=sample)
            )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in simplified_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n====== lm-eval 样本预览保存完成 ======")
    print(f"预览文件: {OUTPUT_PATH}")
    print(f"样本数: {len(simplified_records)}")

    if simplified_records:
        print("\n====== 第一条样本字段预览 ======")
        print(json.dumps(simplified_records[0], ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()