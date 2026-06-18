import csv
import json
import re
from pathlib import Path
from fractions import Fraction


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SFT_SAMPLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "eval"
    / "sft_lora_small_v2_qwen25_15b_gsm8k_cot_limit20"
    / "outputs__checkpoints__sft_lora_small_v2"
)

GRPO_384_SAMPLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "eval"
    / "grpo_lora_small_v2_format_reward_384_qwen25_15b_gsm8k_cot_limit20"
    / "outputs__checkpoints__grpo_lora_small_v2_format_reward_384"
)

OUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUT_CSV = OUT_DIR / "sft_v2_vs_grpo_384_sample_comparison.csv"
OUT_MD = OUT_DIR / "sft_v2_vs_grpo_384_sample_comparison.md"

NUMBER_RE = r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?"


def latest_sample_file(sample_dir: Path) -> Path:
    if not sample_dir.exists():
        raise FileNotFoundError(f"sample_dir not found: {sample_dir}")

    files = sorted(
        sample_dir.glob("samples_gsm8k_cot_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not files:
        raise FileNotFoundError(f"no samples_gsm8k_cot_*.jsonl in: {sample_dir}")

    return files[0]


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
            for key in ["text", "content", "answer", "prediction", "filtered", "exact_match"]:
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


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def classify(sft_ok, grpo_ok):
    if sft_ok == 1 and grpo_ok == 1:
        return "both_correct"
    if sft_ok == 0 and grpo_ok == 0:
        return "both_wrong"
    if sft_ok == 0 and grpo_ok == 1:
        return "grpo_improved"
    if sft_ok == 1 and grpo_ok == 0:
        return "grpo_regressed"
    return "unknown"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sft_file = latest_sample_file(SFT_SAMPLE_DIR)
    grpo_file = latest_sample_file(GRPO_384_SAMPLE_DIR)

    print("====== 精确 sample 文件核对 ======")
    print(f"sft_lora_small_v2: {rel(sft_file)}")
    print(f"grpo_lora_small_v2_format_reward_384: {rel(grpo_file)}")

    sft_samples = load_samples(sft_file)
    grpo_samples = load_samples(grpo_file)

    doc_ids = sorted(set(sft_samples.keys()) | set(grpo_samples.keys()))

    rows = []

    for doc_id in doc_ids:
        sft = sft_samples.get(doc_id, {})
        grpo = grpo_samples.get(doc_id, {})

        gold = sft.get("gold") or grpo.get("gold")
        sft_pred = sft.get("pred")
        grpo_pred = grpo.get("pred")
        sft_ok = sft.get("correct")
        grpo_ok = grpo.get("correct")

        rows.append(
            {
                "doc_id": doc_id,
                "gold": gold,
                "sft_pred": sft_pred,
                "sft_correct": sft_ok,
                "grpo_384_pred": grpo_pred,
                "grpo_384_correct": grpo_ok,
                "category": classify(sft_ok, grpo_ok),
            }
        )

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "doc_id",
                "gold",
                "sft_pred",
                "sft_correct",
                "grpo_384_pred",
                "grpo_384_correct",
                "category",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    sft_correct = sum(1 for row in rows if row["sft_correct"] == 1)
    grpo_correct = sum(1 for row in rows if row["grpo_384_correct"] == 1)

    category_counts = {}
    for row in rows:
        category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1

    lines = []
    lines.append("# SFT small_v2 vs GRPO 384 sample comparison")
    lines.append("")
    lines.append("## 实验目的")
    lines.append("")
    lines.append("本报告逐题对比 `sft_lora_small_v2` 和 `grpo_lora_small_v2_format_reward_384` 的 lm-eval sample 输出。")
    lines.append("")
    lines.append("二者官方 lm-eval 指标相同：")
    lines.append("")
    lines.append("```text")
    lines.append("flexible-extract = 0.6000")
    lines.append("strict-match     = 0.2000")
    lines.append("```")
    lines.append("")
    lines.append("本报告用于判断它们是逐题完全一致，还是只是总分一致。")
    lines.append("")

    lines.append("## 精确 sample 文件核对")
    lines.append("")
    lines.append("| model | sample_file |")
    lines.append("|---|---|")
    lines.append(f"| sft_lora_small_v2 | {rel(sft_file)} |")
    lines.append(f"| grpo_lora_small_v2_format_reward_384 | {rel(grpo_file)} |")
    lines.append("")

    lines.append("## 准确率汇总")
    lines.append("")
    lines.append("| model | correct | total | acc |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| sft_lora_small_v2 | {sft_correct} | {total} | {sft_correct / total if total else 0.0:.4f} |")
    lines.append(f"| grpo_lora_small_v2_format_reward_384 | {grpo_correct} | {total} | {grpo_correct / total if total else 0.0:.4f} |")
    lines.append("")

    lines.append("## 分类汇总")
    lines.append("")
    lines.append("| category | count |")
    lines.append("|---|---:|")
    for category, count in sorted(category_counts.items()):
        lines.append(f"| {category} | {count} |")
    lines.append("")

    lines.append("## 结论解释")
    lines.append("")
    lines.append("- `both_correct` 表示两个模型都答对。")
    lines.append("- `both_wrong` 表示两个模型都答错。")
    lines.append("- `grpo_improved` 表示 SFT-v2 错，但 GRPO-384 对。")
    lines.append("- `grpo_regressed` 表示 SFT-v2 对，但 GRPO-384 错。")
    lines.append("")
    lines.append("如果 `grpo_improved` 和 `grpo_regressed` 都为 0，说明两者逐题表现完全一致。")
    lines.append("如果二者都大于 0，说明总分相同但样本分布有变化。")
    lines.append("")

    lines.append("## Case table")
    lines.append("")
    lines.append("| doc_id | gold | sft_pred | sft_ok | grpo_384_pred | grpo_384_ok | category |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| {row['doc_id']} | {row['gold']} | {row['sft_pred']} | {row['sft_correct']} | "
            f"{row['grpo_384_pred']} | {row['grpo_384_correct']} | {row['category']} |"
        )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("====== sample comparison done ======")
    print(f"csv: {OUT_CSV}")
    print(f"md:  {OUT_MD}")

    print("====== summary ======")
    print(f"sft_lora_small_v2: {sft_correct}/{total} = {sft_correct / total if total else 0.0:.4f}")
    print(
        f"grpo_lora_small_v2_format_reward_384: "
        f"{grpo_correct}/{total} = {grpo_correct / total if total else 0.0:.4f}"
    )

    print("====== category counts ======")
    for category, count in sorted(category_counts.items()):
        print(f"{category}: {count}")


if __name__ == "__main__":
    main()