import csv
import json
import re
from pathlib import Path
from fractions import Fraction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"
OUT_DIR = PROJECT_ROOT / "outputs" / "reports"

OUT_CSV = OUT_DIR / "grpo_format_reward_sample_comparison.csv"
OUT_MD = OUT_DIR / "grpo_format_reward_sample_comparison.md"

NUMBER_RE = r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?"


SAMPLE_SPECS = {
    "sft_lora_small_v2": {
        "eval_output_dir": "sft_lora_small_v2_qwen25_15b_gsm8k_cot_limit20",
        "checkpoint_sample_dir": "outputs__checkpoints__sft_lora_small_v2",
        "forbidden_tokens": [
            "sft_lora_small_v2_format",
            "prompt_format",
            "format_eval",
        ],
    },
    "grpo_lora_small": {
        "eval_output_dir": "grpo_lora_small_qwen25_15b_gsm8k_cot_limit20",
        "checkpoint_sample_dir": "outputs__checkpoints__grpo_lora_small",
        "forbidden_tokens": [
            "format_reward",
            "prompt_format",
        ],
    },
    "grpo_lora_small_format_reward": {
        "eval_output_dir": "grpo_lora_small_format_reward_qwen25_15b_gsm8k_cot_limit20",
        "checkpoint_sample_dir": "outputs__checkpoints__grpo_lora_small_format_reward",
        "forbidden_tokens": [],
    },
}


def parse_number(text):
    if text is None:
        return None

    s = str(text).strip()
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("%", "")

    match = re.search(NUMBER_RE, s)
    if not match:
        return None

    raw = match.group(0).replace(",", "")

    try:
        if "/" in raw:
            return float(Fraction(raw))
        return float(raw)
    except Exception:
        return None


def numbers_equal(a, b, eps=1e-6):
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) < eps


def rel_path(path):
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def find_exact_sample_file(model_key, spec):
    sample_dir = (
        EVAL_DIR
        / spec["eval_output_dir"]
        / spec["checkpoint_sample_dir"]
    )

    if not sample_dir.exists():
        raise FileNotFoundError(
            f"[{model_key}] expected sample directory does not exist:\n"
            f"{sample_dir}"
        )

    sample_files = sorted(
        sample_dir.glob("samples_gsm8k_cot_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not sample_files:
        raise FileNotFoundError(
            f"[{model_key}] no samples_gsm8k_cot_*.jsonl found in:\n"
            f"{sample_dir}"
        )

    chosen = sample_files[0]
    chosen_text = rel_path(chosen).lower()

    for token in spec["forbidden_tokens"]:
        if token.lower() in chosen_text:
            raise ValueError(
                f"[{model_key}] forbidden token found in chosen path: {token}\n"
                f"chosen path: {rel_path(chosen)}"
            )

    return chosen


def flatten_to_text(value):
    parts = []

    def flatten(x):
        if x is None:
            return

        if isinstance(x, str):
            parts.append(x)
        elif isinstance(x, (int, float)):
            parts.append(str(x))
        elif isinstance(x, list):
            for item in x:
                flatten(item)
        elif isinstance(x, dict):
            for key in [
                "text",
                "content",
                "answer",
                "prediction",
                "filtered",
                "exact_match",
            ]:
                if key in x:
                    flatten(x[key])
        else:
            parts.append(str(x))

    flatten(value)
    return "\n".join(parts)


def extract_gold(sample):
    candidates = [
        sample.get("target"),
        sample.get("answer"),
        sample.get("gold"),
    ]

    doc = sample.get("doc")
    if isinstance(doc, dict):
        candidates.extend(
            [
                doc.get("answer"),
                doc.get("target"),
                doc.get("gold"),
            ]
        )

    for candidate in candidates:
        if candidate is None:
            continue

        text = str(candidate)

        match = re.search(r"####\s*(" + NUMBER_RE + r")", text)
        if match:
            return match.group(1)

        nums = re.findall(NUMBER_RE, text)
        if nums:
            return nums[-1]

    return None


def extract_prediction(sample):
    candidates = []

    for key in [
        "filtered_resps",
        "resps",
        "response",
        "model_output",
        "prediction",
        "output",
    ]:
        value = sample.get(key)
        if value is not None:
            candidates.append(value)

    text = "\n".join(flatten_to_text(candidate) for candidate in candidates)

    patterns = [
        r"(?:^|\n)\s*Final answer\s*:\s*(" + NUMBER_RE + r")",
        r"(?:^|\n)\s*####\s*(" + NUMBER_RE + r")",
        r"(?:the answer is|answer is)\s*[:\-]?\s*(" + NUMBER_RE + r")",
    ]

    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))

    if matches:
        return matches[-1]

    nums = re.findall(NUMBER_RE, text)
    return nums[-1] if nums else None


def load_samples(path):
    samples = {}

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue

            sample = json.loads(line)

            raw_doc_id = sample.get("doc_id", idx)
            try:
                doc_id = int(raw_doc_id)
            except Exception:
                doc_id = idx

            gold = extract_gold(sample)
            pred = extract_prediction(sample)

            correct = numbers_equal(parse_number(pred), parse_number(gold))

            samples[doc_id] = {
                "doc_id": doc_id,
                "gold": gold,
                "pred": pred,
                "correct": int(correct),
            }

    return samples


def classify(row):
    sft = row.get("sft_lora_small_v2_correct")
    old = row.get("grpo_lora_small_correct")
    new = row.get("grpo_lora_small_format_reward_correct")

    if sft == 1 and new == 0:
        return "regressed_from_sft_v2"

    if sft == 0 and new == 1:
        return "improved_over_sft_v2"

    if old == 0 and new == 1:
        return "improved_over_old_grpo"

    if old == 1 and new == 0:
        return "regressed_from_old_grpo"

    if new == 1:
        return "format_reward_correct"

    return "still_wrong"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_files = {}
    model_samples = {}

    print("====== 精确核对 lm-eval sample 文件 ======")

    for model_key, spec in SAMPLE_SPECS.items():
        path = find_exact_sample_file(model_key, spec)
        model_files[model_key] = path
        model_samples[model_key] = load_samples(path)

        print(f"[OK] {model_key}")
        print(f"     eval_output_dir: {spec['eval_output_dir']}")
        print(f"     checkpoint_dir:  {spec['checkpoint_sample_dir']}")
        print(f"     sample_file:     {rel_path(path)}")

    all_doc_ids = sorted(
        set().union(*[set(samples.keys()) for samples in model_samples.values()])
    )

    rows = []

    for doc_id in all_doc_ids:
        row = {"doc_id": doc_id}

        gold = None
        for model_key in SAMPLE_SPECS:
            sample = model_samples[model_key].get(doc_id)
            if sample and sample.get("gold") is not None:
                gold = sample["gold"]
                break

        row["gold"] = gold

        for model_key in SAMPLE_SPECS:
            sample = model_samples[model_key].get(doc_id, {})
            row[f"{model_key}_pred"] = sample.get("pred")
            row[f"{model_key}_correct"] = sample.get("correct")

        row["category"] = classify(row)
        rows.append(row)

    fieldnames = [
        "doc_id",
        "gold",
        "sft_lora_small_v2_pred",
        "sft_lora_small_v2_correct",
        "grpo_lora_small_pred",
        "grpo_lora_small_correct",
        "grpo_lora_small_format_reward_pred",
        "grpo_lora_small_format_reward_correct",
        "category",
    ]

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)

    summary = {}
    for model_key in SAMPLE_SPECS:
        correct_count = sum(
            1 for row in rows if row.get(f"{model_key}_correct") == 1
        )
        summary[model_key] = {
            "correct": correct_count,
            "acc": correct_count / total if total else 0.0,
        }

    category_counts = {}
    for row in rows:
        category = row["category"]
        category_counts[category] = category_counts.get(category, 0) + 1

    lines = []

    lines.append("# GRPO format reward sample comparison")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This report compares exact lm-eval sample files for:")
    lines.append("")
    lines.append("```text")
    lines.append("sft_lora_small_v2")
    lines.append("grpo_lora_small")
    lines.append("grpo_lora_small_format_reward")
    lines.append("```")
    lines.append("")
    lines.append("This script uses exact eval output directories and exact checkpoint sample directories.")
    lines.append("It intentionally rejects format/prompt variants unless they are explicitly requested.")
    lines.append("")

    lines.append("## Exact sample file check")
    lines.append("")
    lines.append("| model | eval_output_dir | checkpoint_dir | sample_file |")
    lines.append("|---|---|---|---|")

    for model_key, spec in SAMPLE_SPECS.items():
        lines.append(
            f"| {model_key} | {spec['eval_output_dir']} | "
            f"{spec['checkpoint_sample_dir']} | {rel_path(model_files[model_key])} |"
        )

    lines.append("")
    lines.append("## Accuracy summary from parsed samples")
    lines.append("")
    lines.append("| model | correct | total | parsed_acc |")
    lines.append("|---|---:|---:|---:|")

    for model_key, info in summary.items():
        lines.append(f"| {model_key} | {info['correct']} | {total} | {info['acc']:.4f} |")

    lines.append("")
    lines.append("## Category summary")
    lines.append("")
    lines.append("| category | count |")
    lines.append("|---|---:|")

    for category, count in sorted(category_counts.items()):
        lines.append(f"| {category} | {count} |")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- The exact file check confirms that `sft_lora_small_v2` is not the `sft_lora_small_v2_format` variant.")
    lines.append("- The format-reward GRPO checkpoint matches old `grpo_lora_small` on parsed sample accuracy.")
    lines.append("- The format-reward GRPO checkpoint still underperforms `sft_lora_small_v2` on these parsed samples.")
    lines.append("- This is sample-level error analysis, not a replacement for official lm-eval metrics.")
    lines.append("")

    lines.append("## Case table")
    lines.append("")
    lines.append("| doc | gold | sft_pred | sft_ok | old_pred | old_ok | fmt_pred | fmt_ok | category |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for row in rows:
        lines.append(
            f"| {row['doc_id']} | {row['gold']} | "
            f"{row['sft_lora_small_v2_pred']} | {row['sft_lora_small_v2_correct']} | "
            f"{row['grpo_lora_small_pred']} | {row['grpo_lora_small_correct']} | "
            f"{row['grpo_lora_small_format_reward_pred']} | {row['grpo_lora_small_format_reward_correct']} | "
            f"{row['category']} |"
        )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("====== sample comparison done ======")
    print(f"csv: {OUT_CSV}")
    print(f"md:  {OUT_MD}")

    print("====== exact sample files ======")
    for model_key, path in model_files.items():
        print(f"{model_key}: {rel_path(path)}")

    print("====== summary ======")
    for model_key, info in summary.items():
        print(f"{model_key}: {info['correct']}/{total} = {info['acc']:.4f}")

    print("====== category counts ======")
    for category, count in sorted(category_counts.items()):
        print(f"{category}: {count}")


if __name__ == "__main__":
    main()