import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "eval"
REPORT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_JSON_PATH = EVAL_DIR / "final_rag_eval_summary.json"
SUMMARY_MD_PATH = REPORT_DIR / "final_rag_eval_summary.md"


FILES = {
    "bm25_child": EVAL_DIR / "bm25_child_metrics.json",
    "parent_bm25_child_expansion": EVAL_DIR / "parent_bm25_child_expansion_metrics.json",
    "embedding_child": EVAL_DIR / "embedding_child_metrics.json",
    "hybrid_rrf_default": EVAL_DIR / "hybrid_rrf_metrics.json",
    "hybrid_rrf_best": EVAL_DIR / "hybrid_rrf_best_metrics.json",
    "bge_rerank_hybrid": EVAL_DIR / "bge_rerank_hybrid_metrics.json",
    "final_rag_context_pack": EVAL_DIR / "final_rag_context_metrics.json",
    "rag_answer_generation": EVAL_DIR / "rag_answer_generation_metrics.json",
    "rag_triad_proxy": EVAL_DIR / "rag_triad_proxy_metrics.json",
    "local_llm_judge": EVAL_DIR / "local_llm_judge_metrics.json",
}


def read_json_if_exists(path: Path):
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value):
    if value is None:
        return "-"

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def pick(data, *keys):
    current = data

    for key in keys:
        if current is None:
            return None

        if not isinstance(current, dict):
            return None

        current = current.get(key)

    return current


def main():
    loaded = {}

    print("====== 加载 eval 指标文件 ======")

    for name, path in FILES.items():
        data = read_json_if_exists(path)
        loaded[name] = data

        if data is None:
            print(f"missing: {name} -> {path}")
        else:
            print(f"loaded: {name} -> {path}")

    summary = {
        "project": "hierarchical_rag_agent_eval",
        "loaded_files": {
            name: str(path.relative_to(PROJECT_ROOT))
            for name, path in FILES.items()
            if loaded.get(name) is not None
        },
        "retrieval_summary": {},
        "generation_summary": {},
        "diagnostic_summary": {},
        "final_conclusions": [],
    }

    retrieval_rows = []

    bm25 = loaded.get("bm25_child")
    if bm25:
        retrieval_rows.append({
            "stage": "BM25 child",
            "hit@10": bm25.get("hit@10"),
            "recall@10": bm25.get("recall@10"),
            "mrr@10": bm25.get("mrr@10"),
            "note": "Sparse lexical retrieval baseline.",
        })

    parent = loaded.get("parent_bm25_child_expansion")
    if parent:
        retrieval_rows.append({
            "stage": "Parent BM25 + child expansion",
            "hit@10": parent.get("expanded_child_hit@parent10"),
            "recall@10": parent.get("expanded_child_recall@parent10"),
            "mrr@10": parent.get("expanded_child_mrr@parent10"),
            "note": "Hierarchical parent retrieval followed by child expansion.",
        })

    embedding = loaded.get("embedding_child")
    if embedding:
        retrieval_rows.append({
            "stage": "BGE embedding child",
            "hit@10": embedding.get("hit@10"),
            "recall@10": embedding.get("recall@10"),
            "mrr@10": embedding.get("mrr@10"),
            "note": "Dense semantic retrieval.",
        })

    hybrid_default = loaded.get("hybrid_rrf_default")
    if hybrid_default:
        retrieval_rows.append({
            "stage": "Default Hybrid RRF",
            "hit@10": hybrid_default.get("hit@10"),
            "recall@10": hybrid_default.get("recall@10"),
            "mrr@10": hybrid_default.get("mrr@10"),
            "note": "Initial weighted RRF fusion.",
        })

    hybrid_best = loaded.get("hybrid_rrf_best")
    if hybrid_best:
        retrieval_rows.append({
            "stage": "Best Hybrid RRF",
            "hit@10": hybrid_best.get("hit@10"),
            "recall@10": hybrid_best.get("recall@10"),
            "mrr@10": hybrid_best.get("mrr@10"),
            "note": "Grid-searched RRF weights.",
        })

    rerank = loaded.get("bge_rerank_hybrid")
    if rerank:
        retrieval_rows.append({
            "stage": "Best Hybrid RRF + BGE rerank",
            "hit@10": rerank.get("hit@10"),
            "recall@10": rerank.get("recall@10"),
            "mrr@10": rerank.get("mrr@10"),
            "note": "Final retrieval pipeline.",
        })

    summary["retrieval_summary"]["rows"] = retrieval_rows

    context_pack = loaded.get("final_rag_context_pack")
    generation = loaded.get("rag_answer_generation")
    triad = loaded.get("rag_triad_proxy")
    judge = loaded.get("local_llm_judge")

    if context_pack:
        summary["generation_summary"]["context_pack"] = {
            "context_hit@5": context_pack.get("context_hit@5"),
            "context_recall@5": context_pack.get("context_recall@5"),
            "context_precision@5": context_pack.get("context_precision@5"),
            "context_mrr@5": context_pack.get("context_mrr@5"),
            "context_hit@10": context_pack.get("context_hit@10"),
            "context_recall@10": context_pack.get("context_recall@10"),
            "context_precision@10": context_pack.get("context_precision@10"),
            "context_mrr@10": context_pack.get("context_mrr@10"),
        }

    if generation:
        summary["generation_summary"]["answer_generation"] = {
            "top5_exact_match": pick(generation, "top5", "exact_match"),
            "top5_contains_match": pick(generation, "top5", "contains_match"),
            "top5_avg_context_recall": pick(generation, "top5", "avg_context_recall"),
            "top10_exact_match": pick(generation, "top10", "exact_match"),
            "top10_contains_match": pick(generation, "top10", "contains_match"),
            "top10_avg_context_recall": pick(generation, "top10", "avg_context_recall"),
        }

    if triad:
        summary["diagnostic_summary"]["rag_triad_proxy"] = {
            "top5_groundedness_proxy": pick(triad, "top5", "groundedness_proxy"),
            "top5_answerability_proxy": pick(triad, "top5", "context_answerability_proxy"),
            "top5_soft_triad_pass": pick(triad, "top5", "soft_triad_pass"),
            "top5_error_categories": pick(triad, "top5", "error_category_counts"),
            "top10_groundedness_proxy": pick(triad, "top10", "groundedness_proxy"),
            "top10_answerability_proxy": pick(triad, "top10", "context_answerability_proxy"),
            "top10_soft_triad_pass": pick(triad, "top10", "soft_triad_pass"),
            "top10_error_categories": pick(triad, "top10", "error_category_counts"),
        }

    if judge:
        summary["diagnostic_summary"]["local_llm_judge"] = {
            "top5_llm_correctness": pick(judge, "top5", "llm_judge_answer_correctness"),
            "top5_llm_groundedness": pick(judge, "top5", "llm_judge_groundedness"),
            "top5_error_types": pick(judge, "top5", "error_type_counts"),
            "top10_llm_correctness": pick(judge, "top10", "llm_judge_answer_correctness"),
            "top10_llm_groundedness": pick(judge, "top10", "llm_judge_groundedness"),
            "top10_error_types": pick(judge, "top10", "error_type_counts"),
            "note": "The local 0.5B judge is used as a low-cost evaluation demo and should not be treated as a final authority.",
        }

    summary["final_conclusions"] = [
        "Retrieval optimization is effective: BM25 child Recall@10 improves to the final Best Hybrid RRF + BGE rerank Recall@10 of 0.9467.",
        "Reranking improves evidence ordering: the final retrieval pipeline reaches MRR@10 of 0.9640.",
        "Final context answerability is high, but answer generation with Qwen2.5-0.5B remains weak, indicating the bottleneck shifts from retrieval to generation.",
        "Top10 has higher context recall than Top5, but also lower precision and more noise.",
        "Local LLM-as-a-Judge is useful as a low-cost evaluation framework demo, but the 0.5B judge is not reliable enough for final authority-level evaluation.",
    ]

    with SUMMARY_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("# Final Hierarchical RAG Evaluation Summary")
    lines.append("")
    lines.append("## 1. Retrieval Summary")
    lines.append("")
    lines.append("| Stage | Hit@10 | Recall@10 | MRR@10 | Note |")
    lines.append("|---|---:|---:|---:|---|")

    for row in retrieval_rows:
        lines.append(
            f"| {row['stage']} | "
            f"{fmt(row['hit@10'])} | "
            f"{fmt(row['recall@10'])} | "
            f"{fmt(row['mrr@10'])} | "
            f"{row['note']} |"
        )

    lines.append("")
    lines.append("## 2. Final RAG Context Pack")
    lines.append("")

    cp = summary["generation_summary"].get("context_pack", {})
    lines.append("| Setting | Hit | Recall | Precision | MRR |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| Top5 | {fmt(cp.get('context_hit@5'))} | {fmt(cp.get('context_recall@5'))} | "
        f"{fmt(cp.get('context_precision@5'))} | {fmt(cp.get('context_mrr@5'))} |"
    )
    lines.append(
        f"| Top10 | {fmt(cp.get('context_hit@10'))} | {fmt(cp.get('context_recall@10'))} | "
        f"{fmt(cp.get('context_precision@10'))} | {fmt(cp.get('context_mrr@10'))} |"
    )

    lines.append("")
    lines.append("## 3. Answer Generation Summary")
    lines.append("")

    ag = summary["generation_summary"].get("answer_generation", {})
    lines.append("| Setting | Exact Match | Contains Match | Avg Context Recall |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| Top5 | {fmt(ag.get('top5_exact_match'))} | "
        f"{fmt(ag.get('top5_contains_match'))} | "
        f"{fmt(ag.get('top5_avg_context_recall'))} |"
    )
    lines.append(
        f"| Top10 | {fmt(ag.get('top10_exact_match'))} | "
        f"{fmt(ag.get('top10_contains_match'))} | "
        f"{fmt(ag.get('top10_avg_context_recall'))} |"
    )

    lines.append("")
    lines.append("## 4. RAG Triad Proxy Summary")
    lines.append("")

    triad_summary = summary["diagnostic_summary"].get("rag_triad_proxy", {})
    lines.append("| Setting | Groundedness Proxy | Answerability Proxy | Soft Triad Pass |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| Top5 | {fmt(triad_summary.get('top5_groundedness_proxy'))} | "
        f"{fmt(triad_summary.get('top5_answerability_proxy'))} | "
        f"{fmt(triad_summary.get('top5_soft_triad_pass'))} |"
    )
    lines.append(
        f"| Top10 | {fmt(triad_summary.get('top10_groundedness_proxy'))} | "
        f"{fmt(triad_summary.get('top10_answerability_proxy'))} | "
        f"{fmt(triad_summary.get('top10_soft_triad_pass'))} |"
    )

    lines.append("")
    lines.append("### Top10 Error Categories")
    lines.append("")
    top10_errors = triad_summary.get("top10_error_categories") or {}
    for key, value in sorted(top10_errors.items()):
        lines.append(f"- {key}: `{value}`")

    lines.append("")
    lines.append("## 5. Local LLM-as-a-Judge Summary")
    lines.append("")

    local_judge = summary["diagnostic_summary"].get("local_llm_judge", {})
    lines.append("| Setting | LLM Correctness | LLM Groundedness | Note |")
    lines.append("|---|---:|---:|---|")
    lines.append(
        f"| Top5 | {fmt(local_judge.get('top5_llm_correctness'))} | "
        f"{fmt(local_judge.get('top5_llm_groundedness'))} | "
        f"0.5B local judge, approximate only |"
    )
    lines.append(
        f"| Top10 | {fmt(local_judge.get('top10_llm_correctness'))} | "
        f"{fmt(local_judge.get('top10_llm_groundedness'))} | "
        f"0.5B local judge, approximate only |"
    )

    lines.append("")
    lines.append("## 6. Final Conclusions")
    lines.append("")

    for item in summary["final_conclusions"]:
        lines.append(f"- {item}")

    lines.append("")

    SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("====== Final RAG Eval Summary 完成 ======")
    print("summary_json:", SUMMARY_JSON_PATH)
    print("summary_md:", SUMMARY_MD_PATH)

    print("====== 核心结论 ======")
    for item in summary["final_conclusions"]:
        print("-", item)


if __name__ == "__main__":
    main()