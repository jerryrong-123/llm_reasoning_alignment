import csv
import json
import re
from pathlib import Path


EVAL_ROOT = Path("outputs/eval")
REPORT_DIR = Path("outputs/reports")
BAD_CASE_DIR = Path("outputs/bad_cases")

CSV_PATH = REPORT_DIR / "small_eval_error_analysis.csv"
MD_PATH = REPORT_DIR / "small_eval_error_analysis.md"
BAD_CASE_PATH = BAD_CASE_DIR / "small_eval_bad_cases.jsonl"


SMALL_STAGES = [
    "sft_lora_small",
    "dpo_lora_small",
    "grpo_lora_small",
]


def infer_stage(path: Path) -> str:
    text = str(path).replace("\\", "/").lower()

    if "sft_lora_small" in text:
        return "sft_lora_small"
    if "dpo_lora_small" in text:
        return "dpo_lora_small"
    if "grpo_lora_small" in text:
        return "grpo_lora_small"

    return "unknown"


def load_jsonl(path: Path):
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

    return rows


def extract_gold_answer(sample: dict) -> str:
    target = sample.get("target")
    if target is not None:
        return str(target).strip()

    doc = sample.get("doc", {})
    if not isinstance(doc, dict):
        return ""

    answer = str(doc.get("answer", "")).strip()

    if "####" in answer:
        return answer.split("####")[-1].strip()

    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer)
    if numbers:
        return numbers[-1]

    return answer


def extract_response(sample: dict) -> str:
    """
    lm-eval 的 resps 通常长这样：
    "resps": [["model output"]]

    strict-match 下 filtered_resps 可能是 ["[invalid]"]，
    所以分析模型原始输出时优先用 resps。
    """
    resps = sample.get("resps")

    if isinstance(resps, list) and resps:
        first = resps[0]

        if isinstance(first, list) and first:
            return str(first[0])

        return str(first)

    filtered = sample.get("filtered_resps")

    if isinstance(filtered, list) and filtered:
        return str(filtered[0])

    if filtered is not None:
        return str(filtered)

    return ""


def extract_pred_answer(text: str) -> str:
    """
    从模型输出中粗略抽取最终答案，供人工分析使用。
    真正准确率以 lm-eval 的 exact_match 为准。
    """
    if text is None:
        return ""

    text = str(text)

    patterns = [
        r"[Tt]herefore,\s*the answer is\s*\$?\s*(-?\d+(?:\.\d+)?)",
        r"[Tt]he answer is\s*\$?\s*(-?\d+(?:\.\d+)?)",
        r"[Aa]nswer[:：]\s*\$?\s*(-?\d+(?:\.\d+)?)",
        r"####\s*(-?\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].strip()

    return ""


def to_bool_exact_match(value):
    if value is None:
        return None

    try:
        return float(value) == 1.0
    except Exception:
        return None


def collect_small_samples():
    """
    lm-eval samples 中 strict-match 和 flexible-extract 是分开的行。
    所以这里按 stage + doc_id 合并。
    """
    sample_files = sorted(EVAL_ROOT.rglob("samples_*.jsonl"))

    merged = {}

    for sample_file in sample_files:
        stage = infer_stage(sample_file)

        if stage not in SMALL_STAGES:
            continue

        samples = load_jsonl(sample_file)

        for sample in samples:
            doc_id = sample.get("doc_id")
            filter_name = str(sample.get("filter", "")).strip()
            exact_match = to_bool_exact_match(sample.get("exact_match"))

            key = (stage, doc_id)

            if key not in merged:
                doc = sample.get("doc", {})
                question = doc.get("question", "") if isinstance(doc, dict) else ""
                response = extract_response(sample)

                merged[key] = {
                    "stage": stage,
                    "doc_id": doc_id,
                    "question": question,
                    "gold_answer": extract_gold_answer(sample),
                    "pred_answer": extract_pred_answer(response),
                    "flexible_correct": None,
                    "strict_correct": None,
                    "response": response,
                    "source_file": str(sample_file),
                }

            if filter_name == "flexible-extract":
                merged[key]["flexible_correct"] = exact_match
            elif filter_name == "strict-match":
                merged[key]["strict_correct"] = exact_match

            current_response = merged[key].get("response", "")
            new_response = extract_response(sample)

            if (not current_response or current_response == "[invalid]") and new_response:
                merged[key]["response"] = new_response
                merged[key]["pred_answer"] = extract_pred_answer(new_response)

    rows = list(merged.values())

    rows.sort(
        key=lambda r: (
            SMALL_STAGES.index(r["stage"]) if r["stage"] in SMALL_STAGES else 99,
            int(r["doc_id"] or 0),
        )
    )

    return rows


def summarize(rows):
    summary = {}

    for stage in SMALL_STAGES:
        stage_rows = [r for r in rows if r["stage"] == stage]

        total = len(stage_rows)

        flexible_known = [r for r in stage_rows if r["flexible_correct"] is not None]
        strict_known = [r for r in stage_rows if r["strict_correct"] is not None]

        flexible_correct = sum(1 for r in flexible_known if r["flexible_correct"])
        strict_correct = sum(1 for r in strict_known if r["strict_correct"])

        summary[stage] = {
            "total": total,
            "flexible_known": len(flexible_known),
            "flexible_correct": flexible_correct,
            "flexible_acc": flexible_correct / len(flexible_known) if flexible_known else None,
            "strict_known": len(strict_known),
            "strict_correct": strict_correct,
            "strict_acc": strict_correct / len(strict_known) if strict_known else None,
        }

    return summary


def write_csv(rows):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "stage",
        "doc_id",
        "question",
        "gold_answer",
        "pred_answer",
        "flexible_correct",
        "strict_correct",
        "response",
        "source_file",
    ]

    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_bad_cases(rows):
    BAD_CASE_DIR.mkdir(parents=True, exist_ok=True)

    bad_rows = [
        row for row in rows
        if row["flexible_correct"] is False or row["strict_correct"] is False
    ]

    with BAD_CASE_PATH.open("w", encoding="utf-8") as f:
        for row in bad_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(rows, summary):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Small Stage Error Analysis")
    lines.append("")
    lines.append(
        "> 注意：当前分析基于 lm-eval small 阶段 samples，样本数较小，只用于定位工程链路和输出问题。"
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Stage | Total | Flexible Correct | Flexible Acc | Strict Correct | Strict Acc |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for stage in SMALL_STAGES:
        item = summary[stage]
        flexible_acc = "None" if item["flexible_acc"] is None else f"{item['flexible_acc']:.4f}"
        strict_acc = "None" if item["strict_acc"] is None else f"{item['strict_acc']:.4f}"

        lines.append(
            f"| {stage} | {item['total']} | {item['flexible_correct']} / {item['flexible_known']} | "
            f"{flexible_acc} | {item['strict_correct']} / {item['strict_known']} | {strict_acc} |"
        )

    lines.append("")
    lines.append("## Bad Case Preview")
    lines.append("")
    lines.append(
        "下面每个 small 阶段最多展示 5 个 bad cases。"
        "其中 flexible_correct=False 表示答案抽取后仍错误；"
        "strict_correct=False 可能包含格式问题，例如答案正确但不符合 strict-match 格式。"
    )
    lines.append("")

    for stage in SMALL_STAGES:
        lines.append(f"### {stage}")
        lines.append("")

        stage_bad_rows = [
            row for row in rows
            if row["stage"] == stage
            and (row["flexible_correct"] is False or row["strict_correct"] is False)
        ]

        if not stage_bad_rows:
            lines.append("当前阶段没有 bad cases。")
            lines.append("")
            continue

        for row in stage_bad_rows[:5]:
            lines.append(f"#### doc_id {row['doc_id']}")
            lines.append("")
            lines.append(f"- Gold answer: `{row['gold_answer']}`")
            lines.append(f"- Pred answer: `{row['pred_answer']}`")
            lines.append(f"- Flexible correct: `{row['flexible_correct']}`")
            lines.append(f"- Strict correct: `{row['strict_correct']}`")
            lines.append("")
            lines.append("Question:")
            lines.append("")
            lines.append(row["question"])
            lines.append("")
            lines.append("Model response:")
            lines.append("")
            lines.append("```text")
            lines.append(str(row["response"])[:1200])
            lines.append("```")
            lines.append("")

    lines.append("## Preliminary Observations")
    lines.append("")
    lines.append(
        "- 如果 `Pred answer` 与 `Gold answer` 相同，但 `Strict correct=False`，"
        "说明主要是输出格式没有满足 strict-match。"
    )
    lines.append(
        "- 如果 `Pred answer` 与 `Gold answer` 不同，说明是数学推理、题意理解或计算过程错误。"
    )
    lines.append(
        "- 当前 DPO small 和 GRPO small 的 flexible / strict 指标没有超过 SFT small，"
        "需要后续继续分析偏好数据质量、reward 稀疏性、训练步数和 prompt / 输出格式。"
    )
    lines.append("")

    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows = collect_small_samples()

    if not rows:
        print("没有找到 small 阶段 lm-eval samples。")
        return

    summary = summarize(rows)

    write_csv(rows)
    write_bad_cases(rows)
    write_markdown(rows, summary)

    print("====== small 阶段样本错误分析完成 ======")
    print(f"CSV 文件: {CSV_PATH}")
    print(f"Markdown 文件: {MD_PATH}")
    print(f"Bad cases 文件: {BAD_CASE_PATH}")

    print("")
    print("====== 汇总预览 ======")
    for stage, item in summary.items():
        print(stage, item)


if __name__ == "__main__":
    main()