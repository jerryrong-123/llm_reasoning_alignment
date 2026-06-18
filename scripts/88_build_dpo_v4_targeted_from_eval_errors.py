import json
from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dpo_v4_targeted_from_sft_grpo_errors.jsonl"
PREVIEW_PATH = PROJECT_ROOT / "data" / "samples" / "dpo_v4_targeted_from_sft_grpo_errors_preview.jsonl"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "dpo_v4_targeted_from_sft_grpo_errors_report.md"

SOURCE_PATTERNS = {
    "sft_v5_limit200": "outputs/eval/sft_lora_v5_eval_style_completion_only_qwen25_15b_gsm8k_cot_limit200/**/samples_gsm8k_cot_*.jsonl",
    "grpo_v6_quick50_limit200": "outputs/eval/grpo_lora_v6_quick_conservative_50_qwen25_15b_gsm8k_cot_limit200/**/samples_gsm8k_cot_*.jsonl",
    "grpo_v9_ckpt50_limit200": "outputs/eval/grpo_v9_checkpoint-50_qwen25_15b_gsm8k_cot_limit200/**/samples_gsm8k_cot_*.jsonl",
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def get_model_response(obj):
    """
    lm-eval 的 resps 通常是 list[list[str]]，例如 [["模型完整输出"]]
    这里取完整生成文本，不取 filtered_resps，因为 filtered_resps 通常只是抽取后的答案。
    """
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


def normalize_gold_answer(answer: str):
    """
    GSM8K answer 一般是：
    推理过程
    #### 最终答案

    DPO chosen 直接保留官方完整解法，有利于学习正确推理和最终答案格式。
    """
    return str(answer).strip()


def build_prompt(question: str):
    """
    保持和 eval-style SFT 接近的 prompt 格式。
    """
    return f"Q: {question.strip()}\nA:"


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_records = []
    source_stats = {}

    seen_pairs = set()

    print("====== 构造 DPO_v4 targeted 数据 ======")

    for source_name, pattern in SOURCE_PATTERNS.items():
        paths = sorted(PROJECT_ROOT.glob(pattern))
        print(f"\n[source] {source_name}")
        print(f"pattern: {pattern}")
        print(f"matched files: {len(paths)}")

        if not paths:
            source_stats[source_name] = {
                "files": 0,
                "flexible_rows": 0,
                "wrong_rows": 0,
                "added": 0,
            }
            continue

        # 同一个 source 一般只会匹配一个最新 samples 文件；如果多个，就全部读
        flexible_rows = 0
        wrong_rows = 0
        added = 0

        for path in paths:
            print(f"读取: {path.relative_to(PROJECT_ROOT)}")

            for obj in read_jsonl(path):
                # 只用 flexible-extract 判断答案是否正确
                # strict-match 主要看格式，不能作为 DPO 错题判断依据
                if obj.get("filter") != "flexible-extract":
                    continue

                flexible_rows += 1

                exact = float(obj.get("exact_match", 0.0))
                if exact >= 1.0:
                    continue

                wrong_rows += 1

                doc = obj.get("doc", {})
                question = str(doc.get("question", "")).strip()
                gold_answer = str(doc.get("answer", "")).strip()
                target = str(obj.get("target", "")).strip()
                rejected = get_model_response(obj)

                if not question or not gold_answer or not rejected:
                    continue

                prompt = build_prompt(question)
                chosen = normalize_gold_answer(gold_answer)

                # 过滤太短/明显坏的 rejected
                if len(rejected) < 5:
                    continue

                pair_key = (source_name, obj.get("doc_id"), rejected[:300])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                rec = {
                    "source": source_name,
                    "doc_id": obj.get("doc_id"),
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "target": target,
                    "question": question,
                    "gold_answer": gold_answer,
                }

                all_records.append(rec)
                added += 1

        source_stats[source_name] = {
            "files": len(paths),
            "flexible_rows": flexible_rows,
            "wrong_rows": wrong_rows,
            "added": added,
        }

    # 打乱不要做，先保持可复现顺序
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with PREVIEW_PATH.open("w", encoding="utf-8") as f:
        for rec in all_records[:20]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_source = defaultdict(int)
    for rec in all_records:
        by_source[rec["source"]] += 1

    report_lines = []
    report_lines.append("# DPO_v4 Targeted Preference Data Report")
    report_lines.append("")
    report_lines.append("## Purpose")
    report_lines.append("")
    report_lines.append("This dataset is built from evaluation error cases of SFT_v5 and GRPO models. For each wrong model output, the official GSM8K solution is used as chosen and the model's incorrect generation is used as rejected.")
    report_lines.append("")
    report_lines.append("## Input Sources")
    report_lines.append("")
    for source_name, stats in source_stats.items():
        report_lines.append(f"- {source_name}: files={stats['files']}, flexible_rows={stats['flexible_rows']}, wrong_rows={stats['wrong_rows']}, added={stats['added']}")
    report_lines.append("")
    report_lines.append("## Output")
    report_lines.append("")
    report_lines.append(f"- train_file: `{OUTPUT_PATH.relative_to(PROJECT_ROOT)}`")
    report_lines.append(f"- preview_file: `{PREVIEW_PATH.relative_to(PROJECT_ROOT)}`")
    report_lines.append(f"- total_records: {len(all_records)}")
    report_lines.append("")
    report_lines.append("## Counts by Source")
    report_lines.append("")
    for source_name, count in by_source.items():
        report_lines.append(f"- {source_name}: {count}")
    report_lines.append("")
    report_lines.append("## Fields")
    report_lines.append("")
    report_lines.append("- prompt: eval-style question prompt")
    report_lines.append("- chosen: official GSM8K gold solution")
    report_lines.append("- rejected: incorrect model generation")
    report_lines.append("- target: final numeric answer")
    report_lines.append("- source: model/eval source")
    report_lines.append("- doc_id: GSM8K sample id in lm-eval")
    report_lines.append("")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n====== 完成 ======")
    print(f"输出 DPO 数据: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"预览文件: {PREVIEW_PATH.relative_to(PROJECT_ROOT)}")
    print(f"报告文件: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"总样本数: {len(all_records)}")
    print("\n====== source 统计 ======")
    for source_name, stats in source_stats.items():
        print(source_name, stats)


if __name__ == "__main__":
    main()
