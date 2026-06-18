import json
from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dpo_v6_same_style_correct_vs_wrong.jsonl"
PREVIEW_PATH = PROJECT_ROOT / "data" / "samples" / "dpo_v6_same_style_correct_vs_wrong_preview.jsonl"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "dpo_v6_same_style_correct_vs_wrong_report.md"

SOURCE_PATTERNS = {
    "sft_v5_limit200": "outputs/eval/sft_lora_v5_eval_style_completion_only_qwen25_15b_gsm8k_cot_limit200/**/samples_gsm8k_cot_*.jsonl",
    "grpo_v6_quick50_limit200": "outputs/eval/grpo_lora_v6_quick_conservative_50_qwen25_15b_gsm8k_cot_limit200/**/samples_gsm8k_cot_*.jsonl",
    "grpo_v9_ckpt50_limit200": "outputs/eval/grpo_v9_checkpoint-50_qwen25_15b_gsm8k_cot_limit200/**/samples_gsm8k_cot_*.jsonl",
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_model_response(obj):
    resps = obj.get("resps", [])
    if isinstance(resps, list) and resps:
        first = resps[0]
        if isinstance(first, list) and first:
            return str(first[0]).strip()
        return str(first).strip()

    filtered = obj.get("filtered_resps", [])
    if isinstance(filtered, list) and filtered:
        return str(filtered[0]).strip()

    return ""


def build_prompt(question: str):
    return f"Q: {question.strip()}\nA:"


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows_by_doc = defaultdict(list)
    source_stats = {}

    for source_name, pattern in SOURCE_PATTERNS.items():
        paths = sorted(PROJECT_ROOT.glob(pattern))
        print(f"\n[source] {source_name}")
        print("matched files:", len(paths))

        flexible_rows = 0
        correct_rows = 0
        wrong_rows = 0

        for path in paths:
            print("读取:", path.relative_to(PROJECT_ROOT))

            for obj in read_jsonl(path):
                if obj.get("filter") != "flexible-extract":
                    continue

                flexible_rows += 1

                doc = obj.get("doc", {})
                question = str(doc.get("question", "")).strip()
                gold_answer = str(doc.get("answer", "")).strip()
                target = str(obj.get("target", "")).strip()
                response = get_model_response(obj)
                exact = float(obj.get("exact_match", 0.0))

                if not question or not response:
                    continue

                if exact >= 1.0:
                    correct_rows += 1
                else:
                    wrong_rows += 1

                rows_by_doc[obj.get("doc_id")].append({
                    "source": source_name,
                    "doc_id": obj.get("doc_id"),
                    "question": question,
                    "gold_answer": gold_answer,
                    "target": target,
                    "response": response,
                    "exact": exact,
                })

        source_stats[source_name] = {
            "files": len(paths),
            "flexible_rows": flexible_rows,
            "correct_rows": correct_rows,
            "wrong_rows": wrong_rows,
        }

    records = []
    seen = set()

    for doc_id, rows in sorted(rows_by_doc.items(), key=lambda x: x[0]):
        correct_rows = [r for r in rows if r["exact"] >= 1.0 and len(r["response"]) >= 10]
        wrong_rows = [r for r in rows if r["exact"] < 1.0 and len(r["response"]) >= 10]

        if not correct_rows or not wrong_rows:
            continue

        chosen_r = correct_rows[0]
        rejected_r = wrong_rows[0]

        key = (doc_id, chosen_r["response"][:200], rejected_r["response"][:200])
        if key in seen:
            continue
        seen.add(key)

        records.append({
            "source": f"{chosen_r['source']}_chosen__{rejected_r['source']}_rejected",
            "pair_type": "same_style_correct_vs_wrong",
            "doc_id": doc_id,
            "prompt": build_prompt(chosen_r["question"]),
            "chosen": chosen_r["response"],
            "rejected": rejected_r["response"],
            "target": chosen_r["target"],
            "question": chosen_r["question"],
            "gold_answer": chosen_r["gold_answer"],
        })

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with PREVIEW_PATH.open("w", encoding="utf-8") as f:
        for rec in records[:20]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = []
    report.append("# DPO_v6 Same-style Correct-vs-Wrong Report")
    report.append("")
    report.append("## Source Stats")
    for k, v in source_stats.items():
        report.append(f"- {k}: {v}")
    report.append("")
    report.append("## Output")
    report.append(f"- train_file: `{OUTPUT_PATH.relative_to(PROJECT_ROOT)}`")
    report.append(f"- preview_file: `{PREVIEW_PATH.relative_to(PROJECT_ROOT)}`")
    report.append(f"- total_records: {len(records)}")
    report.append("")
    report.append("## Meaning")
    report.append("")
    report.append("chosen is a model-generated correct response, rejected is a model-generated wrong response. This avoids forcing the model to imitate gold-answer style directly.")

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print("\n====== 完成 ======")
    print("输出:", OUTPUT_PATH.relative_to(PROJECT_ROOT))
    print("预览:", PREVIEW_PATH.relative_to(PROJECT_ROOT))
    print("报告:", REPORT_PATH.relative_to(PROJECT_ROOT))
    print("总样本数:", len(records))


if __name__ == "__main__":
    main()
