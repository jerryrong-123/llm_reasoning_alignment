import json
from pathlib import Path


BASELINE_DIR = Path("outputs/eval/baseline_qwen25_15b_gsm8k_cot_limit200")
SFT_V5_DIR = Path("outputs/eval/sft_lora_v5_eval_style_completion_only_qwen25_15b_gsm8k_cot_limit200")

OUTPUT_DIR = Path("outputs/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "baseline_vs_sft_v5_limit200_sample_comparison.json"


def find_sample_file(eval_dir: Path) -> Path:
    files = list(eval_dir.glob("**/*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No jsonl sample file found under {eval_dir}")
    return files[0]


def load_flexible_results(sample_file: Path):
    results = {}

    with sample_file.open("r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)

            if x.get("filter") != "flexible-extract":
                continue

            doc_id = x["doc_id"]
            doc = x.get("doc", {})
            target = str(x.get("target", "")).strip()
            exact = float(x.get("exact_match", 0.0)) > 0

            results[doc_id] = {
                "doc_id": doc_id,
                "question": doc.get("question", ""),
                "target": target,
                "exact_match": exact,
                "filtered_resps": x.get("filtered_resps"),
                "resps": x.get("resps"),
            }

    return results


def unwrap_response(resps):
    x = resps
    while isinstance(x, list) and len(x) > 0:
        x = x[0]
    return str(x)


def main():
    baseline_file = find_sample_file(BASELINE_DIR)
    sft_file = find_sample_file(SFT_V5_DIR)

    baseline = load_flexible_results(baseline_file)
    sft = load_flexible_results(sft_file)

    common_ids = sorted(set(baseline) & set(sft))

    both_correct = []
    both_wrong = []
    sft_improved = []
    sft_regressed = []

    for doc_id in common_ids:
        b = baseline[doc_id]
        s = sft[doc_id]

        b_ok = b["exact_match"]
        s_ok = s["exact_match"]

        row = {
            "doc_id": doc_id,
            "question": b["question"],
            "target": b["target"],
            "baseline_correct": b_ok,
            "sft_v5_correct": s_ok,
            "baseline_filtered": b["filtered_resps"],
            "sft_v5_filtered": s["filtered_resps"],
            "baseline_response_head": unwrap_response(b["resps"])[:600],
            "sft_v5_response_head": unwrap_response(s["resps"])[:600],
        }

        if b_ok and s_ok:
            both_correct.append(row)
        elif (not b_ok) and (not s_ok):
            both_wrong.append(row)
        elif (not b_ok) and s_ok:
            sft_improved.append(row)
        elif b_ok and (not s_ok):
            sft_regressed.append(row)

    result = {
        "baseline_sample_file": str(baseline_file),
        "sft_v5_sample_file": str(sft_file),
        "total_common": len(common_ids),
        "baseline_correct": sum(1 for x in baseline.values() if x["exact_match"]),
        "sft_v5_correct": sum(1 for x in sft.values() if x["exact_match"]),
        "baseline_accuracy": sum(1 for x in baseline.values() if x["exact_match"]) / len(common_ids),
        "sft_v5_accuracy": sum(1 for x in sft.values() if x["exact_match"]) / len(common_ids),
        "both_correct": len(both_correct),
        "both_wrong": len(both_wrong),
        "sft_v5_improved": len(sft_improved),
        "sft_v5_regressed": len(sft_regressed),
        "net_gain": len(sft_improved) - len(sft_regressed),
        "improved_examples": sft_improved[:10],
        "regressed_examples": sft_regressed[:10],
    }

    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("baseline_sample_file:", baseline_file)
    print("sft_v5_sample_file:", sft_file)
    print("total_common:", result["total_common"])
    print("baseline_correct:", result["baseline_correct"])
    print("sft_v5_correct:", result["sft_v5_correct"])
    print("baseline_accuracy:", result["baseline_accuracy"])
    print("sft_v5_accuracy:", result["sft_v5_accuracy"])
    print("both_correct:", result["both_correct"])
    print("both_wrong:", result["both_wrong"])
    print("sft_v5_improved:", result["sft_v5_improved"])
    print("sft_v5_regressed:", result["sft_v5_regressed"])
    print("net_gain:", result["net_gain"])
    print("saved_to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
