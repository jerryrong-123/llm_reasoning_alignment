import json
from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dpo_v5_mixed_conservative.jsonl"
PREVIEW_PATH = PROJECT_ROOT / "data" / "samples" / "dpo_v5_mixed_conservative_preview.jsonl"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "dpo_v5_mixed_conservative_report.md"

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

    print("====== 读取 SFT_v5 / GRPO_v6 / GRPO_v9 limit200 samples ======")

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

                if not question or not gold_answer or not response:
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

    hard_pairs = []
    safe_pairs = []
    seen = set()

    print("\n====== 构造 hard pairs: gold chosen vs wrong rejected ======")

    for doc_id, rows in sorted(rows_by_doc.items(), key=lambda x: x[0]):
        for r in rows:
            if r["exact"] >= 1.0:
                continue

            key = ("hard", r["source"], doc_id, r["response"][:200])
            if key in seen:
                continue
            seen.add(key)

            hard_pairs.append({
                "pair_type": "hard_gold_vs_wrong",
                "source": r["source"],
                "doc_id": doc_id,
                "prompt": build_prompt(r["question"]),
                "chosen": r["gold_answer"],
                "rejected": r["response"],
                "target": r["target"],
                "question": r["question"],
                "gold_answer": r["gold_answer"],
            })

    print("hard_pairs:", len(hard_pairs))

    print("\n====== 构造 safe pairs: correct model output vs wrong model output ======")

    for doc_id, rows in sorted(rows_by_doc.items(), key=lambda x: x[0]):
        correct_rows = [r for r in rows if r["exact"] >= 1.0 and len(r["response"]) >= 5]
        wrong_rows = [r for r in rows if r["exact"] < 1.0 and len(r["response"]) >= 5]

        if not correct_rows or not wrong_rows:
            continue

        # 每道题最多加 1 条 safe pair，避免某几道题重复太多
        chosen_r = correct_rows[0]
        rejected_r = wrong_rows[0]

        key = ("safe", doc_id, chosen_r["response"][:200], rejected_r["response"][:200])
        if key in seen:
            continue
        seen.add(key)

        safe_pairs.append({
            "pair_type": "safe_correct_vs_wrong",
            "source": f"{chosen_r['source']}_chosen__{rejected_r['source']}_rejected",
            "doc_id": doc_id,
            "prompt": build_prompt(chosen_r["question"]),
            "chosen": chosen_r["response"],
            "rejected": rejected_r["response"],
            "target": chosen_r["target"],
            "question": chosen_r["question"],
            "gold_answer": chosen_r["gold_answer"],
        })

    print("safe_pairs:", len(safe_pairs))

    # 保守混合：
    # 只取一部分 hard pairs，避免像 DPO_v4 那样全是 hard negative。
    # safe pairs 更接近模型原本输出风格，破坏性更小。
    max_hard = 120
    max_safe = 80

    final_records = []
    final_records.extend(hard_pairs[:max_hard])
    final_records.extend(safe_pairs[:max_safe])

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for rec in final_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with PREVIEW_PATH.open("w", encoding="utf-8") as f:
        for rec in final_records[:20]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    pair_type_counts = defaultdict(int)
    for rec in final_records:
        pair_type_counts[rec["pair_type"]] += 1

    report = []
    report.append("# DPO_v5 Mixed Conservative Data Report")
    report.append("")
    report.append("## Input Source Stats")
    report.append("")
    for source_name, stats in source_stats.items():
        report.append(f"- {source_name}: {stats}")
    report.append("")
    report.append("## Pair Construction")
    report.append("")
    report.append(f"- raw_hard_pairs: {len(hard_pairs)}")
    report.append(f"- raw_safe_pairs: {len(safe_pairs)}")
    report.append(f"- selected_max_hard: {max_hard}")
    report.append(f"- selected_max_safe: {max_safe}")
    report.append("")
    report.append("## Final Counts")
    report.append("")
    report.append(f"- total_records: {len(final_records)}")
    for k, v in pair_type_counts.items():
        report.append(f"- {k}: {v}")
    report.append("")
    report.append("## Output")
    report.append("")
    report.append(f"- train_file: `{OUTPUT_PATH.relative_to(PROJECT_ROOT)}`")
    report.append(f"- preview_file: `{PREVIEW_PATH.relative_to(PROJECT_ROOT)}`")

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print("\n====== 完成 ======")
    print("输出:", OUTPUT_PATH.relative_to(PROJECT_ROOT))
    print("预览:", PREVIEW_PATH.relative_to(PROJECT_ROOT))
    print("报告:", REPORT_PATH.relative_to(PROJECT_ROOT))
    print("总样本数:", len(final_records))
    print("pair_type_counts:", dict(pair_type_counts))


if __name__ == "__main__":
    main()
