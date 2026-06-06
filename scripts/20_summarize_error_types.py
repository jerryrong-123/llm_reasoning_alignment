import csv
from collections import Counter, defaultdict
from pathlib import Path


INPUT_CSV = Path("outputs/reports/small_eval_error_analysis.csv")
REPORT_DIR = Path("outputs/reports")

OUTPUT_CSV = REPORT_DIR / "small_eval_error_type_summary.csv"
OUTPUT_MD = REPORT_DIR / "small_eval_error_type_summary.md"


def parse_bool(value):
    text = str(value).strip().lower()

    if text == "true":
        return True

    if text == "false":
        return False

    return None


def classify_error(row):
    flexible = parse_bool(row.get("flexible_correct"))
    strict = parse_bool(row.get("strict_correct"))

    if flexible is True and strict is True:
        return "correct"

    if flexible is True and strict is False:
        return "format_only_error"

    if flexible is False:
        return "reasoning_or_calc_error"

    return "unknown"


def load_rows():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"找不到输入文件: {INPUT_CSV}。请先运行 scripts/19_analyze_small_eval_samples.py"
        )

    with INPUT_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def summarize(rows):
    stage_counter = defaultdict(Counter)

    for row in rows:
        stage = row.get("stage", "unknown")
        error_type = classify_error(row)
        stage_counter[stage][error_type] += 1

    return stage_counter


def write_csv(stage_counter):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "stage",
        "error_type",
        "count",
        "ratio",
    ]

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for stage, counter in stage_counter.items():
            total = sum(counter.values())

            for error_type, count in counter.items():
                writer.writerow(
                    {
                        "stage": stage,
                        "error_type": error_type,
                        "count": count,
                        "ratio": count / total if total else 0,
                    }
                )


def write_markdown(stage_counter):
    lines = []

    lines.append("# Small Evaluation Error Type Summary")
    lines.append("")
    lines.append(
        "> 本报告基于 small_eval_error_analysis.csv，"
        "用于区分 small 阶段错误到底是格式问题，还是数学推理 / 计算问题。"
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Stage | Error Type | Count | Ratio |")
    lines.append("|---|---|---:|---:|")

    for stage, counter in stage_counter.items():
        total = sum(counter.values())

        for error_type in [
            "correct",
            "format_only_error",
            "reasoning_or_calc_error",
            "unknown",
        ]:
            count = counter.get(error_type, 0)
            ratio = count / total if total else 0

            lines.append(
                f"| {stage} | {error_type} | {count} | {ratio:.4f} |"
            )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- `format_only_error`：答案数值已经对了，但 strict-match 不通过，说明需要优化输出格式。"
    )
    lines.append(
        "- `reasoning_or_calc_error`：答案数值本身错了，说明是题意理解、推理链或计算错误。"
    )
    lines.append(
        "- 如果 format_only_error 较多，下一步应优先做格式约束实验。"
    )
    lines.append(
        "- 如果 reasoning_or_calc_error 较多，下一步应优先改进训练数据、DPO 数据质量、reward 或训练步数。"
    )
    lines.append("")

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows = load_rows()
    stage_counter = summarize(rows)

    write_csv(stage_counter)
    write_markdown(stage_counter)

    print("====== small 错误类型汇总完成 ======")
    print(f"CSV 文件: {OUTPUT_CSV}")
    print(f"Markdown 文件: {OUTPUT_MD}")

    print("")
    print("====== 汇总预览 ======")
    for stage, counter in stage_counter.items():
        print(stage, dict(counter))


if __name__ == "__main__":
    main()