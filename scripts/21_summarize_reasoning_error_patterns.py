import csv
from collections import Counter, defaultdict
from pathlib import Path


INPUT_CSV = Path("outputs/reports/small_eval_error_analysis.csv")
REPORT_DIR = Path("outputs/reports")

OUTPUT_CSV = REPORT_DIR / "small_reasoning_error_pattern_summary.csv"
OUTPUT_MD = REPORT_DIR / "small_reasoning_error_pattern_summary.md"


PATTERN_ORDER = [
    "percentage_error",
    "unit_rate_error",
    "money_profit_error",
    "multi_step_relation_error",
    "counting_quantity_error",
    "unknown_reasoning_error",
]


def parse_bool(value):
    text = str(value).strip().lower()

    if text == "true":
        return True

    if text == "false":
        return False

    return None


def normalize_answer(value):
    if value is None:
        return ""

    text = str(value).strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.strip()

    import re
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


def is_reasoning_or_calc_error(row):
    flexible = parse_bool(row.get("flexible_correct"))

    if flexible is not False:
        return False

    # 如果 pred_answer 和 gold_answer 实际相等，
    # 说明更像答案抽取 / 格式问题，不算真正 reasoning error。
    if pred_equals_gold(row):
        return False

    return True


def classify_reasoning_pattern(row):
    """
    基于关键词的轻量规则分类。
    注意：这是 small 阶段诊断工具，不是严格人工标注。
    后续可以把输出报告作为人工复核入口。
    """
    question = str(row.get("question", "")).lower()
    response = str(row.get("response", "")).lower()
    text = question + "\n" + response

    percentage_keywords = [
        "%",
        "percent",
        "percentage",
        "increased",
        "decreased",
        "increase",
        "decrease",
        "remaining",
    ]

    unit_rate_keywords = [
        "per",
        "rate",
        "mph",
        "gb",
        "minute",
        "hour",
        "hours",
        "miles",
        "meters",
        "cups",
    ]

    money_profit_keywords = [
        "$",
        "cost",
        "price",
        "profit",
        "sells",
        "sold",
        "market",
        "worth",
        "value",
        "dollars",
        "pay",
        "bought",
    ]

    multi_step_keywords = [
        "then",
        "after",
        "before",
        "left",
        "remaining",
        "another",
        "total",
        "start",
        "started",
        "finish",
    ]

    counting_keywords = [
        "dozen",
        "each",
        "every",
        "half",
        "third",
        "twice",
        "times",
        "flock",
        "glasses",
        "eggs",
    ]

    if any(k in text for k in percentage_keywords):
        return "percentage_error"

    if any(k in text for k in money_profit_keywords):
        return "money_profit_error"

    if any(k in text for k in unit_rate_keywords):
        return "unit_rate_error"

    if any(k in text for k in counting_keywords):
        return "counting_quantity_error"

    if any(k in text for k in multi_step_keywords):
        return "multi_step_relation_error"

    return "unknown_reasoning_error"


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
    examples = defaultdict(list)

    for row in rows:
        if not is_reasoning_or_calc_error(row):
            continue

        stage = row.get("stage", "unknown")
        pattern = classify_reasoning_pattern(row)

        stage_counter[stage][pattern] += 1

        if len(examples[(stage, pattern)]) < 3:
            examples[(stage, pattern)].append(row)

    return stage_counter, examples


def write_csv(stage_counter):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "stage",
        "reasoning_error_pattern",
        "count",
        "ratio",
    ]

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for stage, counter in stage_counter.items():
            total = sum(counter.values())

            for pattern in PATTERN_ORDER:
                count = counter.get(pattern, 0)

                writer.writerow(
                    {
                        "stage": stage,
                        "reasoning_error_pattern": pattern,
                        "count": count,
                        "ratio": count / total if total else 0,
                    }
                )


def write_markdown(stage_counter, examples):
    lines = []

    lines.append("# Small Reasoning Error Pattern Summary")
    lines.append("")
    lines.append(
        "> 本报告基于 small_eval_error_analysis.csv，"
        "只分析 flexible_correct=False 的 reasoning_or_calc_error 样本。"
        "当前分类是关键词规则生成的初步诊断，不等同于人工严格标注。"
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Stage | Reasoning Error Pattern | Count | Ratio |")
    lines.append("|---|---|---:|---:|")

    for stage, counter in stage_counter.items():
        total = sum(counter.values())

        for pattern in PATTERN_ORDER:
            count = counter.get(pattern, 0)
            ratio = count / total if total else 0
            lines.append(f"| {stage} | {pattern} | {count} | {ratio:.4f} |")

    lines.append("")
    lines.append("## Example Cases")
    lines.append("")

    for stage, counter in stage_counter.items():
        lines.append(f"### {stage}")
        lines.append("")

        for pattern in PATTERN_ORDER:
            rows = examples.get((stage, pattern), [])

            if not rows:
                continue

            lines.append(f"#### {pattern}")
            lines.append("")

            for row in rows:
                lines.append(f"- doc_id: `{row.get('doc_id')}`")
                lines.append(f"  - Gold answer: `{row.get('gold_answer')}`")
                lines.append(f"  - Pred answer: `{row.get('pred_answer')}`")
                lines.append(f"  - Question: {row.get('question')}")
                lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- 如果 `percentage_error` 多，说明模型在百分比变化、增量和利润计算上容易错。"
    )
    lines.append(
        "- 如果 `unit_rate_error` 多，说明单位、速率、时间、距离等关系需要加强。"
    )
    lines.append(
        "- 如果 `money_profit_error` 多，说明成本、售价、利润、价值变化类题目需要补充训练样本。"
    )
    lines.append(
        "- 如果 `multi_step_relation_error` 多，说明模型在多步状态转移和剩余量计算上不稳定。"
    )
    lines.append(
        "- 如果 `unknown_reasoning_error` 多，说明关键词规则不足，需要人工复核或引入更细标注。"
    )
    lines.append("")

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows = load_rows()
    stage_counter, examples = summarize(rows)

    write_csv(stage_counter)
    write_markdown(stage_counter, examples)

    print("====== small reasoning 错误模式汇总完成 ======")
    print(f"CSV 文件: {OUTPUT_CSV}")
    print(f"Markdown 文件: {OUTPUT_MD}")

    print("")
    print("====== 汇总预览 ======")
    for stage, counter in stage_counter.items():
        print(stage, dict(counter))


if __name__ == "__main__":
    main()