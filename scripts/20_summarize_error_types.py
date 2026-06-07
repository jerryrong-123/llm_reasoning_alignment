import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


INPUT_CSV = Path("outputs/reports/small_eval_error_analysis.csv")
REPORT_DIR = Path("outputs/reports")

OUTPUT_CSV = REPORT_DIR / "small_eval_error_type_summary.csv"
OUTPUT_MD = REPORT_DIR / "small_eval_error_type_summary.md"


ERROR_TYPE_ORDER = [
    "correct",
    "strict_format_only_error",
    "answer_extraction_or_format_error",
    "reasoning_or_calc_error",
    "unknown",
]


def parse_bool(value):
    text = str(value).strip().lower()

    if text == "true":
        return True

    if text == "false":
        return False

    return None


def normalize_answer(value):
    """
    用于比较 pred_answer 和 gold_answer 是否实际相等。

    处理：
    - 逗号：70,000 -> 70000
    - 美元符号：$18 -> 18
    - 空格
    - 整数小数：18.0 -> 18
    """
    if value is None:
        return ""

    text = str(value).strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("％", "%")
    text = text.strip()

    # 只抽取最后一个数字，避免字符串里混入说明文字
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)

    if numbers:
        num = numbers[-1]

        try:
            value_float = float(num)

            if value_float.is_integer():
                return str(int(value_float))

            return str(value_float)
        except Exception:
            return num

    return text.lower()


def pred_equals_gold(row):
    pred = normalize_answer(row.get("pred_answer"))
    gold = normalize_answer(row.get("gold_answer"))

    if not pred or not gold:
        return False

    return pred == gold


def classify_error(row):
    flexible = parse_bool(row.get("flexible_correct"))
    strict = parse_bool(row.get("strict_correct"))

    if flexible is True and strict is True:
        return "correct"

    if flexible is True and strict is False:
        return "strict_format_only_error"

    if flexible is False:
        if pred_equals_gold(row):
            return "answer_extraction_or_format_error"

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

            for error_type in ERROR_TYPE_ORDER:
                count = counter.get(error_type, 0)

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
        "用于区分 small 阶段错误到底是格式问题、答案抽取问题，还是数学推理 / 计算问题。"
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Stage | Error Type | Count | Ratio |")
    lines.append("|---|---|---:|---:|")

    for stage, counter in stage_counter.items():
        total = sum(counter.values())

        for error_type in ERROR_TYPE_ORDER:
            count = counter.get(error_type, 0)
            ratio = count / total if total else 0

            lines.append(
                f"| {stage} | {error_type} | {count} | {ratio:.4f} |"
            )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- `correct`：flexible-extract 和 strict-match 都通过。"
    )
    lines.append(
        "- `strict_format_only_error`：答案数值正确，flexible-extract 通过，但 strict-match 不通过，主要是 strict 输出格式问题。"
    )
    lines.append(
        "- `answer_extraction_or_format_error`：lm-eval 的 flexible-extract 判错，但脚本抽取的 pred_answer 与 gold_answer 实际相等，可能是答案抽取或格式兼容问题。"
    )
    lines.append(
        "- `reasoning_or_calc_error`：pred_answer 与 gold_answer 不一致，说明答案数值本身错误，更可能是题意理解、推理链或计算错误。"
    )
    lines.append(
        "- `unknown`：无法根据当前字段判断。"
    )
    lines.append("")
    lines.append("## Next Step")
    lines.append("")
    lines.append(
        "后续 reasoning 错误模式分析应该只针对 `reasoning_or_calc_error`，"
        "不要把 `answer_extraction_or_format_error` 误算成真正推理错误。"
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