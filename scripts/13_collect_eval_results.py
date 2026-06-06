import json
import csv
from pathlib import Path


EVAL_ROOT = Path("outputs/eval")
REPORT_DIR = Path("outputs/reports")
CSV_PATH = REPORT_DIR / "eval_summary.csv"
MD_PATH = REPORT_DIR / "eval_summary.md"


def infer_stage(path: Path) -> str:
    """
    从 lm-eval 结果路径推断实验阶段。

    注意：
    small 阶段必须优先判断。
    否则 dpo_lora_small 会被 dpo_lora 提前匹配，导致报告阶段名错误。
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


def find_result_json_files():
    """
    lm-eval 不同版本保存结果文件名可能不同。
    所以这里递归扫描 outputs/eval 下所有 json 文件，
    只要里面包含 results 字段，就认为是汇总结果。
    """
    files = []

    for path in EVAL_ROOT.rglob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict) and "results" in data:
                files.append(path)

        except Exception:
            continue

    return files


def extract_rows_from_result(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stage = infer_stage(path)
    rows = []

    results = data.get("results", {})

    for task_name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue

        for metric_key, value in metrics.items():
            if "stderr" in metric_key:
                continue

            if isinstance(value, (int, float)):
                stderr_key = None
                stderr_value = None

                # lm-eval 常见格式：
                # exact_match,flexible-extract
                # exact_match_stderr,flexible-extract
                if "," in metric_key:
                    metric_name, filter_name = metric_key.split(",", 1)
                    candidate_stderr_key = f"{metric_name}_stderr,{filter_name}"
                    stderr_key = candidate_stderr_key
                else:
                    candidate_stderr_key = f"{metric_key}_stderr"
                    stderr_key = candidate_stderr_key

                if stderr_key in metrics:
                    stderr_value = metrics[stderr_key]

                rows.append(
                    {
                        "stage": stage,
                        "task": task_name,
                        "metric": metric_key,
                        "value": value,
                        "stderr": stderr_value,
                        "source_file": str(path),
                    }
                )

    return rows


def write_csv(rows):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = ["stage", "task", "metric", "value", "stderr", "source_file"]

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    preferred_order = {
        "baseline": 0,
        "sft_lora": 1,
        "dpo_lora": 2,
        "grpo_lora": 3,
        "unknown": 99,
    }

    rows = sorted(
        rows,
        key=lambda x: (
            preferred_order.get(x["stage"], 99),
            x["task"],
            x["metric"],
        ),
    )

    lines = []
    lines.append("# Evaluation Summary")
    lines.append("")
    lines.append("> 注意：当前结果来自 debug 设置，例如 limit=5、max_steps=1，只能证明流程跑通，不能作为正式模型性能。")
    lines.append("")
    lines.append("| Stage | Task | Metric | Value | Stderr |")
    lines.append("|---|---|---|---:|---:|")

    for row in rows:
        value = row["value"]
        stderr = row["stderr"]

        value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
        stderr_str = f"{stderr:.4f}" if isinstance(stderr, float) else str(stderr)

        lines.append(
            f"| {row['stage']} | {row['task']} | {row['metric']} | {value_str} | {stderr_str} |"
        )

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("====== 扫描 lm-eval 结果文件 ======")

    result_files = find_result_json_files()

    if not result_files:
        print("没有找到 lm-eval 汇总结果 JSON。")
        print("请确认 outputs/eval/ 下已经有 baseline / sft / dpo / grpo 的评估结果。")
        return

    print("找到结果文件：")
    for path in result_files:
        print("-", path)

    all_rows = []

    for path in result_files:
        all_rows.extend(extract_rows_from_result(path))

    if not all_rows:
        print("找到了 JSON 文件，但没有提取到 metric。")
        return

    write_csv(all_rows)
    write_markdown(all_rows)

    print("\n====== 评估结果汇总完成 ======")
    print(f"CSV 文件: {CSV_PATH}")
    print(f"Markdown 文件: {MD_PATH}")

    print("\n====== 汇总预览 ======")
    for row in all_rows:
        print(
            row["stage"],
            row["task"],
            row["metric"],
            row["value"],
            row["stderr"],
        )


if __name__ == "__main__":
    main()