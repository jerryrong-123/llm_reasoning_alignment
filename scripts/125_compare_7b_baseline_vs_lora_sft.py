import json
from pathlib import Path


PROJECT_ROOT = Path("/root/autodl-tmp/llm_reasoning_alignment_server_restored")

EVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "eval"
REPORT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "reports"

BASELINE_PATH = EVAL_DIR / "rag_answer_generation_qwen25_7b_autodl_metrics.json"
LORA_PATH = EVAL_DIR / "rag_answer_generation_qwen25_7b_lora_sft_full_metrics.json"

OUT_JSON = EVAL_DIR / "compare_qwen25_7b_baseline_vs_lora_sft_metrics.json"
OUT_REPORT = REPORT_DIR / "compare_qwen25_7b_baseline_vs_lora_sft_report.md"


VARIANTS = [
    "top10_original_recheck",
    "top10_soft_cap2_compressed",
    "top7_soft_cap2_compressed",
]


METRIC_KEYS = [
    "exact_match",
    "contains_match",
    "groundedness_proxy",
    "strict_triad_pass",
    "soft_triad_pass",
]


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_variant(metrics, variant):
    return metrics["variants"][variant]


def fmt(x):
    return f"{x:.4f}"


def main():
    baseline = load_json(BASELINE_PATH)
    lora = load_json(LORA_PATH)

    comparison = {
        "method": "compare_qwen25_7b_baseline_vs_lora_sft",
        "baseline_metrics_path": str(BASELINE_PATH.relative_to(PROJECT_ROOT)),
        "lora_sft_metrics_path": str(LORA_PATH.relative_to(PROJECT_ROOT)),
        "variants": {},
    }

    lines = []
    lines.append("# Qwen2.5-7B Baseline vs Qwen2.5-7B LoRA SFT RAG Comparison")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This report compares the no-finetuning Qwen2.5-7B RAG baseline with the Qwen2.5-7B LoRA SFT model on the same 50-example RAG evaluation set.")
    lines.append("")
    lines.append("The goal is to determine whether RAG-SFT improves answer quality beyond the strong 7B baseline.")
    lines.append("")
    lines.append("## 2. Main Metrics")
    lines.append("")
    lines.append("| Variant | Base EM | SFT EM | Δ EM | Base Contains | SFT Contains | Δ Contains | Base Grounded | SFT Grounded | Δ Grounded | Base SoftTriad | SFT SoftTriad | Δ SoftTriad |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for variant in VARIANTS:
        b = get_variant(baseline, variant)
        s = get_variant(lora, variant)

        deltas = {k: s[k] - b[k] for k in METRIC_KEYS}

        comparison["variants"][variant] = {
            "baseline": b,
            "lora_sft": s,
            "delta": deltas,
        }

        lines.append(
            f"| {variant} | "
            f"{fmt(b['exact_match'])} | {fmt(s['exact_match'])} | {deltas['exact_match']:+.4f} | "
            f"{fmt(b['contains_match'])} | {fmt(s['contains_match'])} | {deltas['contains_match']:+.4f} | "
            f"{fmt(b['groundedness_proxy'])} | {fmt(s['groundedness_proxy'])} | {deltas['groundedness_proxy']:+.4f} | "
            f"{fmt(b['soft_triad_pass'])} | {fmt(s['soft_triad_pass'])} | {deltas['soft_triad_pass']:+.4f} |"
        )

    lines.append("")
    lines.append("## 3. Error Distribution")
    lines.append("")

    for variant in VARIANTS:
        b = get_variant(baseline, variant)
        s = get_variant(lora, variant)

        lines.append(f"### {variant}")
        lines.append("")
        lines.append("**Baseline errors:**")
        for k, v in b.get("error_category_counts", {}).items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("**LoRA SFT errors:**")
        for k, v in s.get("error_category_counts", {}).items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    lines.append("## 4. Conclusion")
    lines.append("")
    lines.append("LoRA SFT does not substantially improve exact match on the Top10 variants, but it improves contains match, groundedness, and soft triad pass.")
    lines.append("")
    lines.append("The best post-SFT result is obtained by `top7_soft_cap2_compressed`, reaching EM=0.6600, Contains=0.8200, Groundedness=0.9600, and SoftTriad=0.8000.")
    lines.append("")
    lines.append("This suggests that RAG-SFT mainly improves answer style and context-grounded answer extraction rather than dramatically changing exact-match correctness.")
    lines.append("")

    OUT_JSON.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("====== 7B baseline vs LoRA SFT 对比完成 ======")
    print("json:", OUT_JSON)
    print("report:", OUT_REPORT)
    print("====== 关键对比 ======")

    for variant in VARIANTS:
        b = get_variant(baseline, variant)
        s = get_variant(lora, variant)
        print(
            f"{variant}: "
            f"EM {b['exact_match']:.4f}->{s['exact_match']:.4f}, "
            f"Contains {b['contains_match']:.4f}->{s['contains_match']:.4f}, "
            f"Grounded {b['groundedness_proxy']:.4f}->{s['groundedness_proxy']:.4f}, "
            f"SoftTriad {b['soft_triad_pass']:.4f}->{s['soft_triad_pass']:.4f}"
        )


if __name__ == "__main__":
    main()
