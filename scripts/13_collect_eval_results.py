import csv
import json
from pathlib import Path


EVAL_ROOT = Path("outputs/eval")
REPORT_DIR = Path("outputs/reports")
CSV_PATH = REPORT_DIR / "eval_summary.csv"
MD_PATH = REPORT_DIR / "eval_summary.md"


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
    从 lm-eval 结果路径推断实验阶段。

    注意：
    small 阶段必须优先判断。
    否则 dpo_lora_small 会被 dpo_lora 提前匹配。
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


def get_sample_len(data: dict, task_name: str):
    n_samples = data.get("n-samples", {})
    task_samples = n_samples.get(task_name)

    if isinstance(task_samples, dict):
        return task_samples.get("effective") or task_samples.get("original")

    if isinstance(task_samples, int):
        return task_samples

    return None


def get_stderr_key(metric_key: str) -> str:
    if "," in metric_key:
        name, suffix = metric_key.split(",", 1)
        return f"{name}_stderr,{suffix}"

    return f"{metric_key}_stderr"


def is_metric_key(key: str, value) -> bool:
    if key == "alias":
        return False
    if "stderr" in key:
        return False
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def collect_rows():
    print("====== 扫描 lm-eval 结果文件 ======")

    result_files = sorted(EVAL_ROOT.rglob("results_*.json"))

    if not result_files:
        print("没有找到 lm-eval 结果文件。")
        return []

    print("找到结果文件：")
    for path in result_files:
        print(f"- {path}")

    rows = []

    for result_file in result_files:
        stage = infer_stage(result_file)

        with result_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", {})

        for task_name, metrics in results.items():
            sample_len = get_sample_len(data, task_name)

            rows.append(
                {
                    "stage": stage,
                    "task": task_name,
                    "sample_len": sample_len,
                    "metric": "sample_len",
                    "value": sample_len,
                    "stderr": None,
                    "result_file": str(result_file),
                    "mtime": result_file.stat().st_mtime,
                }
            )

            for metric_key, metric_value in metrics.items():
                if not is_metric_key(metric_key, metric_value):
                    continue

                stderr_key = get_stderr_key(metric_key)
                stderr_value = metrics.get(stderr_key)

                rows.append(
                    {
                        "stage": stage,
                        "task": task_name,
                        "sample_len": sample_len,
                        "metric": metric_key,
                        "value": metric_value,
                        "stderr": stderr_value,
                        "result_file": str(result_file),
                        "mtime": result_file.stat().st_mtime,
                    }
                )

    return deduplicate_latest(rows)


def deduplicate_latest(rows):
    """
    同一个 stage / task / sample_len / metric 如果重复跑过，
    只保留最新一次，避免 baseline、sft_lora 等重复显示。
    """
    latest = {}

    for row in rows:
        key = (
            row["stage"],
            row["task"],
            row["sample_len"],
            row["metric"],
        )

        if key not in latest or row["mtime"] > latest[key]["mtime"]:
            latest[key] = row

    output = list(latest.values())

    output.sort(
        key=lambda r: (
            STAGE_ORDER.get(r["stage"], 99),
            str(r["task"]),
            int(r["sample_len"] or 0),
            str(r["metric"]),
        )
    )

    for row in output:
        row.pop("mtime", None)

    return output


def format_number(value):
    if value is None:
        return "None"

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def write_csv(rows):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "stage",
        "task",
        "sample_len",
        "metric",
        "value",
        "stderr",
        "result_file",
    ]

    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Evaluation Summary")
    lines.append("")
    lines.append(
        "> 注意：当前结果来自 debug / small 设置，例如 limit=5/20、max_steps=1/10/20。"
        "这些结果用于验证 SFT-DPO-GRPO/RLVR 闭环流程和小规模对比，不能作为正式模型性能。"
    )
    lines.append("")
    lines.append("| Stage | Task | Sample Len | Metric | Value | Stderr |")
    lines.append("|---|---|---:|---|---:|---:|")

    for row in rows:
        lines.append(
            "| {stage} | {task} | {sample_len} | {metric} | {value} | {stderr} |".format(
                stage=row["stage"],
                task=row["task"],
                sample_len=row["sample_len"],
                metric=row["metric"],
                value=format_number(row["value"]),
                stderr=format_number(row["stderr"]),
            )
        )

    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows = collect_rows()

    if not rows:
        return

    write_csv(rows)
    write_markdown(rows)

    print("")
    print("====== 评估结果汇总完成 ======")
    print(f"CSV 文件: {CSV_PATH}")
    print(f"Markdown 文件: {MD_PATH}")

    print("")
    print("====== 汇总预览 ======")
    for row in rows:
        print(
            row["stage"],
            row["task"],
            row["sample_len"],
            row["metric"],
            row["value"],
            row["stderr"],
        )


if __name__ == "__main__":
    main()