import csv
import json
import re
from pathlib import Path
from collections import defaultdict


EVAL_DIR = Path("outputs/eval")
REPORT_DIR = Path("outputs/reports")

OUTPUT_CSV = REPORT_DIR / "small_v2_sample_comparison.csv"
OUTPUT_MD = REPORT_DIR / "small_v2_sample_comparison.md"

TARGET_STAGES = [
    "sft_lora_small",
    "sft_lora_small_v2",
    "sft_lora_small_v2_format",
]


def infer_stage(path: Path) -> str:
    text = str(path).replace("\\", "/").lower()

    if "sft_lora_small_v2_format" in text:
        return "sft_lora_small_v2_format"

    if "sft_lora_small_v2" in text:
        return "sft_lora_small_v2"

    if "sft_lora_small" in text:
        return "sft_lora_small"

    return "unknown"


def find_latest_sample_file_by_stage():
    stage_to_files = defaultdict(list)

    for path in EVAL_DIR.rglob("samples_*.jsonl"):
        stage = infer_stage(path)

        if stage in TARGET_STAGES:
            stage_to_files[stage].append(path)

    latest = {}

    for stage, files in stage_to_files.items():
        files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
        latest[stage] = files[0]

    missing = [stage for stage in TARGET_STAGES if stage not in latest]

    if missing:
        raise FileNotFoundError(f"缺少这些 stage 的 samples 文件: {missing}")

    return latest


def normalize_answer(value):
    if value is None:
        return ""

    text = str(value).strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("％", "%")

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

    return text.lower().strip()


def extract_gold_answer(item):
    target = item.get("target")

    if target is not None:
        return normalize_answer(target)

    doc = item.get("doc", {})
    answer = doc.get("answer", "")

    if "####" in str(answer):
        return normalize_answer(str(answer).split("####")[-1])

    return normalize_answer(answer)


def extract_model_response(item):
    resps = item.get("resps", [])

    if isinstance(resps, list) and resps:
        first = resps[0]

        if isinstance(first, list) and first:
            return str(first[0])

        return str(first)

    filtered = item.get("filtered_resps", [])

    if isinstance(filtered, list) and filtered:
        return str(filtered[0])

    return ""


def extract_pred_answer(item):
    filtered = item.get("filtered_resps", [])

    if isinstance(filtered, list) and filtered:
        value = str(filtered[0]).strip()

        if value and value != "[invalid]":
            return normalize_answer(value)

    response = extract_model_response(item)
    return normalize_answer(response)


def get_doc_id(item, fallback_idx):
    if "doc_id" in item:
        return str(item["doc_id"])

    if "doc_hash" in item:
        return str(item["doc_hash"])

    return str(fallback_idx)


def get_filter_name(item):
    return str(item.get("filter", "")).strip()


def get_exact_match(item):
    value = item.get("exact_match")

    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def load_stage_samples(stage, path):
    combined = {}

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            doc_id = get_doc_id(item, idx)
            filter_name = get_filter_name(item)

            doc = item.get("doc", {})
            question = str(doc.get("question", "")).strip()
            gold_answer = extract_gold_answer(item)
            pred_answer = extract_pred_answer(item)
            response = extract_model_response(item)
            exact_match = get_exact_match(item)

            if doc_id not in combined:
                combined[doc_id] = {
                    "doc_id": doc_id,
                    "question": question,
                    "gold_answer": gold_answer,
                    f"{stage}_pred_answer": pred_answer,
                    f"{stage}_response": response,
                    f"{stage}_flexible_correct": None,
                    f"{stage}_strict_correct": None,
                }

            if filter_name == "flexible-extract":
                combined[doc_id][f"{stage}_flexible_correct"] = exact_match == 1.0

            if filter_name == "strict-match":
                combined[doc_id][f"{stage}_strict_correct"] = exact_match == 1.0

            if not combined[doc_id].get(f"{stage}_pred_answer"):
                combined[doc_id][f"{stage}_pred_answer"] = pred_answer

            if not combined[doc_id].get(f"{stage}_response"):
                combined[doc_id][f"{stage}_response"] = response

    return combined


def merge_samples(stage_data):
    all_doc_ids = sorted(
        set().union(*[set(data.keys()) for data in stage_data.values()])
    )

    rows = []

    for doc_id in all_doc_ids:
        row = {
            "doc_id": doc_id,
            "question": "",
            "gold_answer": "",
        }

        for stage in TARGET_STAGES:
            item = stage_data.get(stage, {}).get(doc_id, {})

            if item.get("question") and not row["question"]:
                row["question"] = item["question"]

            if item.get("gold_answer") and not row["gold_answer"]:
                row["gold_answer"] = item["gold_answer"]

            row[f"{stage}_pred_answer"] = item.get(f"{stage}_pred_answer", "")
            row[f"{stage}_flexible_correct"] = item.get(f"{stage}_flexible_correct")
            row[f"{stage}_strict_correct"] = item.get(f"{stage}_strict_correct")
            row[f"{stage}_response"] = item.get(f"{stage}_response", "")

        small = row["sft_lora_small_flexible_correct"]
        v2 = row["sft_lora_small_v2_flexible_correct"]
        fmt = row["sft_lora_small_v2_format_flexible_correct"]

        if small is False and v2 is True:
            row["small_to_v2_change"] = "fixed_by_targeted_v2"
        elif small is True and v2 is False:
            row["small_to_v2_change"] = "regressed_in_targeted_v2"
        elif small == v2:
            row["small_to_v2_change"] = "unchanged"
        else:
            row["small_to_v2_change"] = "unknown"

        if v2 is True and fmt is False:
            row["v2_to_format_change"] = "broken_by_format_constraint"
        elif v2 is False and fmt is True:
            row["v2_to_format_change"] = "fixed_by_format_constraint"
        elif v2 == fmt:
            row["v2_to_format_change"] = "unchanged"
        else:
            row["v2_to_format_change"] = "unknown"

        rows.append(row)

    return rows


def summarize(rows):
    summary = {
        "small_to_v2": defaultdict(int),
        "v2_to_format": defaultdict(int),
    }

    for row in rows:
        summary["small_to_v2"][row["small_to_v2_change"]] += 1
        summary["v2_to_format"][row["v2_to_format_change"]] += 1

    return summary


def write_csv(rows):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "doc_id",
        "question",
        "gold_answer",
        "sft_lora_small_pred_answer",
        "sft_lora_small_flexible_correct",
        "sft_lora_small_strict_correct",
        "sft_lora_small_v2_pred_answer",
        "sft_lora_small_v2_flexible_correct",
        "sft_lora_small_v2_strict_correct",
        "sft_lora_small_v2_format_pred_answer",
        "sft_lora_small_v2_format_flexible_correct",
        "sft_lora_small_v2_format_strict_correct",
        "small_to_v2_change",
        "v2_to_format_change",
    ]

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_markdown(rows, summary, stage_files):
    lines = []

    lines.append("# Small v2 Sample Comparison")
    lines.append("")
    lines.append(
        "> 本报告对比 sft_lora_small、sft_lora_small_v2、"
        "sft_lora_small_v2_format 在同一批 GSM8K-COT limit=20 样本上的表现。"
    )
    lines.append("")

    lines.append("## Source Files")
    lines.append("")

    for stage, path in stage_files.items():
        lines.append(f"- `{stage}`: `{path}`")

    lines.append("")
    lines.append("## Summary")
    lines.append("")

    lines.append("### sft_lora_small -> sft_lora_small_v2")
    lines.append("")
    lines.append("| Change Type | Count |")
    lines.append("|---|---:|")

    for key, count in sorted(summary["small_to_v2"].items()):
        lines.append(f"| {key} | {count} |")

    lines.append("")
    lines.append("### sft_lora_small_v2 -> sft_lora_small_v2_format")
    lines.append("")
    lines.append("| Change Type | Count |")
    lines.append("|---|---:|")

    for key, count in sorted(summary["v2_to_format"].items()):
        lines.append(f"| {key} | {count} |")

    lines.append("")
    lines.append("## Key Cases")
    lines.append("")

    key_case_types = [
        "fixed_by_targeted_v2",
        "regressed_in_targeted_v2",
        "broken_by_format_constraint",
        "fixed_by_format_constraint",
    ]

    for case_type in key_case_types:
        lines.append(f"### {case_type}")
        lines.append("")

        selected = [
            row
            for row in rows
            if row["small_to_v2_change"] == case_type
            or row["v2_to_format_change"] == case_type
        ][:5]

        if not selected:
            lines.append("- None")
            lines.append("")
            continue

        for row in selected:
            lines.append(f"#### doc_id {row['doc_id']}")
            lines.append("")
            lines.append(f"- Gold answer: `{row['gold_answer']}`")
            lines.append(
                f"- sft_lora_small: `{row['sft_lora_small_pred_answer']}` "
                f"flexible={row['sft_lora_small_flexible_correct']}"
            )
            lines.append(
                f"- sft_lora_small_v2: `{row['sft_lora_small_v2_pred_answer']}` "
                f"flexible={row['sft_lora_small_v2_flexible_correct']}"
            )
            lines.append(
                f"- sft_lora_small_v2_format: `{row['sft_lora_small_v2_format_pred_answer']}` "
                f"flexible={row['sft_lora_small_v2_format_flexible_correct']}"
            )
            lines.append("")
            lines.append("Question:")
            lines.append("")
            lines.append(row["question"])
            lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- `fixed_by_targeted_v2` 表示 targeted small_v2 修复了 small 阶段错题。"
    )
    lines.append(
        "- `regressed_in_targeted_v2` 表示 targeted small_v2 让原本答对的题变错。"
    )
    lines.append(
        "- `broken_by_format_constraint` 表示 format 版本破坏了 targeted small_v2 原本答对的题。"
    )
    lines.append(
        "- `fixed_by_format_constraint` 表示 format 版本修复了 targeted small_v2 原本答错的题。"
    )
    lines.append("")

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    print("====== 查找各阶段最新 samples 文件 ======")
    stage_files = find_latest_sample_file_by_stage()

    for stage, path in stage_files.items():
        print(f"{stage}: {path}")

    stage_data = {}

    for stage, path in stage_files.items():
        stage_data[stage] = load_stage_samples(stage, path)

    rows = merge_samples(stage_data)
    summary = summarize(rows)

    write_csv(rows)
    write_markdown(rows, summary, stage_files)

    print("")
    print("====== small_v2 样本对比完成 ======")
    print(f"CSV 文件: {OUTPUT_CSV}")
    print(f"Markdown 文件: {OUTPUT_MD}")

    print("")
    print("====== 汇总预览 ======")
    print("small -> v2:", dict(summary["small_to_v2"]))
    print("v2 -> format:", dict(summary["v2_to_format"]))


if __name__ == "__main__":
    main()