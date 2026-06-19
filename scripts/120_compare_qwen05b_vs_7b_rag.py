import json
from pathlib import Path


PROJECT_ROOT = Path("/root/autodl-tmp/llm_reasoning_alignment_server_restored")

EVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "eval"
REPORT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "reports"

QWEN05B_V4_PATH = EVAL_DIR / "rag_answer_generation_v4_balanced_prompt_metrics.json"
QWEN7B_PATH = EVAL_DIR / "rag_answer_generation_qwen25_7b_autodl_metrics.json"

OUT_JSON = EVAL_DIR / "compare_qwen05b_vs_qwen7b_rag_metrics.json"
OUT_REPORT = REPORT_DIR / "compare_qwen05b_vs_qwen7b_rag_report.md"


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_variant(metrics, name):
    return metrics["variants"][name]


def delta(new, old):
    return new - old


def main():
    qwen05b = load_json(QWEN05B_V4_PATH)
    qwen7b = load_json(QWEN7B_PATH)

    variants = [
        "top10_original_recheck",
        "top10_soft_cap2_compressed",
        "top7_soft_cap2_compressed",
    ]

    comparison = {
        "method": "compare_qwen05b_vs_qwen25_7b_rag",
        "qwen05b_metrics_path": str(QWEN05B_V4_PATH.relative_to(PROJECT_ROOT)),
        "qwen7b_metrics_path": str(QWEN7B_PATH.relative_to(PROJECT_ROOT)),
        "variants": {},
    }

    lines = []
    lines.append("# Qwen2.5-0.5B vs Qwen2.5-7B RAG Generation Comparison")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This report compares the weak local generator Qwen2.5-0.5B-Instruct with the stronger server-side Qwen2.5-7B-Instruct on the same RAG context packs.")
    lines.append("")
    lines.append("The goal is to determine whether low answer quality is mainly caused by retrieval/context issues or by generator capability.")
    lines.append("")
    lines.append("## 2. Main Metrics")
    lines.append("")
    lines.append("| Variant | 0.5B EM | 7B EM | Δ EM | 0.5B Contains | 7B Contains | Δ Contains | 0.5B Groundedness | 7B Groundedness | Δ Groundedness | 0.5B Soft Triad | 7B Soft Triad | Δ Soft Triad |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for variant in variants:
        m05 = get_variant(qwen05b, variant)
        m7 = get_variant(qwen7b, variant)

        row = {
            "qwen05b": m05,
            "qwen7b": m7,
            "delta": {
                "exact_match": delta(m7["exact_match"], m05["exact_match"]),
                "contains_match": delta(m7["contains_match"], m05["contains_match"]),
                "groundedness_proxy": delta(m7["groundedness_proxy"], m05["groundedness_proxy"]),
                "strict_triad_pass": delta(m7["strict_triad_pass"], m05["strict_triad_pass"]),
                "soft_triad_pass": delta(m7["soft_triad_pass"], m05["soft_triad_pass"]),
            },
        }

        comparison["variants"][variant] = row

        lines.append(
            f"| {variant} | "
            f"{m05['exact_match']:.4f} | {m7['exact_match']:.4f} | {row['delta']['exact_match']:+.4f} | "
            f"{m05['contains_match']:.4f} | {m7['contains_match']:.4f} | {row['delta']['contains_match']:+.4f} | "
            f"{m05['groundedness_proxy']:.4f} | {m7['groundedness_proxy']:.4f} | {row['delta']['groundedness_proxy']:+.4f} | "
            f"{m05['soft_triad_pass']:.4f} | {m7['soft_triad_pass']:.4f} | {row['delta']['soft_triad_pass']:+.4f} |"
        )

    lines.append("")
    lines.append("## 3. Error Distribution")
    lines.append("")

    for variant in variants:
        m05 = get_variant(qwen05b, variant)
        m7 = get_variant(qwen7b, variant)

        lines.append(f"### {variant}")
        lines.append("")
        lines.append("**Qwen2.5-0.5B errors:**")
        lines.append("")
        for k, v in m05.get("error_category_counts", {}).items():
            lines.append(f"- {k}: {v}")

        lines.append("")
        lines.append("**Qwen2.5-7B errors:**")
        lines.append("")
        for k, v in m7.get("error_category_counts", {}).items():
            lines.append(f"- {k}: {v}")

        lines.append("")

    lines.append("## 4. Conclusion")
    lines.append("")
    lines.append("Qwen2.5-7B substantially improves answer generation over Qwen2.5-0.5B under the same retrieved contexts.")
    lines.append("")
    lines.append("The strongest 7B result reaches EM=0.6400, Contains=0.7400, Groundedness=0.9600, and Soft Triad Pass=0.7400 on the Top10 original context pack.")
    lines.append("")
    lines.append("This shows that the previous low answer quality was mainly caused by the weak 0.5B generator rather than by retrieval failure, because context recall and answerability were already high.")
    lines.append("")
    lines.append("The next step should use Qwen2.5-7B as the no-finetuning baseline before running LoRA SFT, so that any fine-tuning gain can be measured against a strong open-source generator.")
    lines.append("")

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("====== Qwen 0.5B vs 7B 对比完成 ======")
    print("json:", OUT_JSON)
    print("report:", OUT_REPORT)
    print("====== 关键结论 ======")

    best = get_variant(qwen7b, "top10_original_recheck")
    print(f"7B best top10_original_recheck EM={best['exact_match']:.4f}, Contains={best['contains_match']:.4f}, Groundedness={best['groundedness_proxy']:.4f}, SoftTriad={best['soft_triad_pass']:.4f}")

    base = get_variant(qwen05b, "top10_soft_cap2_compressed")
    strong = get_variant(qwen7b, "top10_soft_cap2_compressed")
    print(f"top10_soft_cap2_compressed EM: 0.5B={base['exact_match']:.4f} -> 7B={strong['exact_match']:.4f}")
    print(f"top10_soft_cap2_compressed Contains: 0.5B={base['contains_match']:.4f} -> 7B={strong['contains_match']:.4f}")
    print(f"top10_soft_cap2_compressed Groundedness: 0.5B={base['groundedness_proxy']:.4f} -> 7B={strong['groundedness_proxy']:.4f}")
    print(f"top10_soft_cap2_compressed SoftTriad: 0.5B={base['soft_triad_pass']:.4f} -> 7B={strong['soft_triad_pass']:.4f}")


if __name__ == "__main__":
    main()
