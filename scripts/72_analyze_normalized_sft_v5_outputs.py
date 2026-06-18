import json
import re
from pathlib import Path


SAMPLE_FILE = Path(
    "outputs/eval/sft_lora_v5_eval_style_completion_only_qwen25_15b_gsm8k_cot_limit100"
).glob("**/*.jsonl")

SAMPLE_FILE = list(SAMPLE_FILE)[0]

OUTPUT_DIR = Path("outputs/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "sft_v5_normalized_answer_analysis.json"


def unwrap_response(resps):
    x = resps
    while isinstance(x, list) and len(x) > 0:
        x = x[0]
    return str(x)


def normalize_number(s):
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace(",", "").replace("$", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return ""
    num = m.group(0)
    if num.endswith(".0"):
        num = num[:-2]
    return num


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
            return normalize_number(matches[-1])

    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if nums:
        return normalize_number(nums[-1])

    return ""


def main():
    total = 0
    official_correct = 0
    normalized_correct = 0
    extracted = 0
    invalid_count = 0
    invalid_to_correct = 0
    official_correct_to_normalized_wrong = 0

    improved_examples = []
    failed_examples = []

    with SAMPLE_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            total += 1

            target = normalize_number(x.get("target"))
            official = float(x.get("exact_match", 0.0)) > 0
            if official:
                official_correct += 1

            filtered = x.get("filtered_resps")
            if filtered == ["[invalid]"]:
                invalid_count += 1

            raw_text = unwrap_response(x.get("resps", ""))
            pred = extract_answer(raw_text)

            if pred:
                extracted += 1

            normalized_ok = pred == target
            if normalized_ok:
                normalized_correct += 1

            if (not official) and normalized_ok:
                invalid_to_correct += 1
                if len(improved_examples) < 8:
                    improved_examples.append(
                        {
                            "doc_id": x.get("doc_id"),
                            "target": target,
                            "normalized_pred": pred,
                            "filtered_resps": filtered,
                            "raw_response_head": raw_text[:800],
                        }
                    )

            if official and not normalized_ok:
                official_correct_to_normalized_wrong += 1
                if len(failed_examples) < 5:
                    failed_examples.append(
                        {
                            "doc_id": x.get("doc_id"),
                            "target": target,
                            "normalized_pred": pred,
                            "filtered_resps": filtered,
                            "raw_response_head": raw_text[:800],
                        }
                    )

    result = {
        "sample_file": str(SAMPLE_FILE),
        "total": total,
        "official_flexible_correct": official_correct,
        "official_flexible_accuracy": official_correct / total,
        "official_invalid_count": invalid_count,
        "normalized_extracted_count": extracted,
        "normalized_extract_rate": extracted / total,
        "normalized_correct": normalized_correct,
        "normalized_accuracy": normalized_correct / total,
        "newly_recovered_correct": invalid_to_correct,
        "official_correct_to_normalized_wrong": official_correct_to_normalized_wrong,
        "improved_examples": improved_examples,
        "failed_examples": failed_examples,
    }

    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("sample_file:", SAMPLE_FILE)
    print("total:", total)
    print("official_flexible_correct:", official_correct)
    print("official_flexible_accuracy:", official_correct / total)
    print("official_invalid_count:", invalid_count)
    print("normalized_extracted_count:", extracted)
    print("normalized_extract_rate:", extracted / total)
    print("normalized_correct:", normalized_correct)
    print("normalized_accuracy:", normalized_correct / total)
    print("newly_recovered_correct:", invalid_to_correct)
    print("official_correct_to_normalized_wrong:", official_correct_to_normalized_wrong)
    print("saved_to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
