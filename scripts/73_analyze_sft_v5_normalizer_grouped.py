import json
import re
from collections import defaultdict
from pathlib import Path


SAMPLE_PATH = list(
    Path("outputs/eval/sft_lora_v5_eval_style_completion_only_qwen25_15b_gsm8k_cot_limit100").glob("**/*.jsonl")
)[0]

OUTPUT_DIR = Path("outputs/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "sft_v5_normalized_answer_analysis_grouped.json"


def unwrap_response(resps):
    x = resps
    while isinstance(x, list) and len(x) > 0:
        x = x[0]
    return str(x)


def norm_num(s):
    s = str(s).strip().replace(",", "").replace("$", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return ""
    x = m.group(0)
    if x.endswith(".0"):
        x = x[:-2]
    return x


def extract_answer(text):
    text = str(text)

    patterns = [
        r"####\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
        r"(?:the answer is|answer is|answer:)\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
        r">>\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
    ]

    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        if matches:
            return norm_num(matches[-1])

    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if nums:
        return norm_num(nums[-1])

    return ""


def main():
    rows = []
    with SAMPLE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    print("sample_path:", SAMPLE_PATH)
    print("raw_rows:", len(rows))

    filter_counts = defaultdict(int)
    for x in rows:
        filter_counts[str(x.get("filter"))] += 1

    print("filter_counts:")
    for k, v in sorted(filter_counts.items()):
        print(f"- {k}: {v}")

    # 每道题只保留一条，用 doc_id 去重。raw response 两个 filter 下通常一样。
    by_doc = {}
    for x in rows:
        doc_id = x["doc_id"]
        if doc_id not in by_doc:
            by_doc[doc_id] = x

    total = len(by_doc)
    normalized_correct = 0
    extracted = 0
    examples_correct = []
    examples_wrong = []

    for doc_id, x in sorted(by_doc.items()):
        target = norm_num(x.get("target"))
        raw_response = unwrap_response(x.get("resps", ""))
        pred = extract_answer(raw_response)

        if pred:
            extracted += 1

        ok = pred == target
        if ok:
            normalized_correct += 1
            if len(examples_correct) < 5:
                examples_correct.append(
                    {
                        "doc_id": doc_id,
                        "target": target,
                        "normalized_pred": pred,
                        "normalized_output": f"#### {pred}",
                        "raw_response_head": raw_response[:500],
                    }
                )
        else:
            if len(examples_wrong) < 5:
                examples_wrong.append(
                    {
                        "doc_id": doc_id,
                        "target": target,
                        "normalized_pred": pred,
                        "normalized_output": f"#### {pred}" if pred else "",
                        "raw_response_head": raw_response[:500],
                    }
                )

    result = {
        "sample_path": str(SAMPLE_PATH),
        "raw_rows": len(rows),
        "unique_doc_count": total,
        "filter_counts": dict(filter_counts),
        "official_lmeval_flexible_accuracy": 0.62,
        "official_lmeval_strict_accuracy": 0.07,
        "normalized_extract_count": extracted,
        "normalized_extract_rate": extracted / total,
        "normalized_correct_count": normalized_correct,
        "normalized_accuracy": normalized_correct / total,
        "normalized_format": "#### <extracted_answer>",
        "examples_correct": examples_correct,
        "examples_wrong": examples_wrong,
    }

    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("unique_doc_count:", total)
    print("normalized_extract_count:", extracted)
    print("normalized_extract_rate:", extracted / total)
    print("normalized_correct_count:", normalized_correct)
    print("normalized_accuracy:", normalized_correct / total)
    print("saved_to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
