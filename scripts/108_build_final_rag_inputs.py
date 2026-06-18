import json
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "hierarchical_rag"
RETRIEVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "retrieval"
EVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "eval"
REPORT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "reports"

GOLDEN_EVAL_PATH = DATA_DIR / "golden_eval_50.jsonl"
RERANK_RESULTS_PATH = RETRIEVAL_DIR / "bge_rerank_hybrid_results.jsonl"
RERANK_METRICS_PATH = EVAL_DIR / "bge_rerank_hybrid_metrics.json"

RAG_INPUTS_TOP5_PATH = DATA_DIR / "rag_inputs_top5.jsonl"
RAG_INPUTS_TOP10_PATH = DATA_DIR / "rag_inputs_top10.jsonl"

CONTEXT_METRICS_PATH = EVAL_DIR / "final_rag_context_metrics.json"
REPORT_PATH = REPORT_DIR / "final_rag_context_pack_report.md"

TOP_K_VALUES = [5, 10]


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")


def build_prompt(query, contexts):
    context_lines = []

    for i, ctx in enumerate(contexts, start=1):
        title = ctx.get("title", "")
        text = ctx.get("text", "")
        context_lines.append(f"[{i}] Title: {title}\n{text}")

    context_block = "\n\n".join(context_lines)

    prompt = (
        "You are a question-answering assistant.\n"
        "Answer the question using only the provided contexts.\n"
        "If the contexts do not contain enough information, say that the answer is not supported by the contexts.\n\n"
        f"Question:\n{query}\n\n"
        f"Contexts:\n{context_block}\n\n"
        "Answer:"
    )

    return prompt


def evaluate_contexts(contexts, gold_chunk_ids):
    gold_set = set(gold_chunk_ids)
    retrieved_ids = [ctx["chunk_id"] for ctx in contexts]
    retrieved_set = set(retrieved_ids)

    hit_count = len(gold_set & retrieved_set)

    hit = 1.0 if hit_count > 0 else 0.0
    recall = hit_count / len(gold_set) if gold_set else 0.0
    precision = hit_count / len(retrieved_set) if retrieved_set else 0.0

    first_hit_rank = None
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold_set:
            first_hit_rank = rank
            break

    mrr = 1.0 / first_hit_rank if first_hit_rank is not None else 0.0

    return {
        "hit": hit,
        "recall": recall,
        "precision": precision,
        "mrr": mrr,
        "hit_count": hit_count,
        "context_count": len(contexts),
        "first_hit_rank": first_hit_rank,
    }


def make_rag_input_row(rerank_row, top_k):
    query_id = rerank_row["query_id"]
    query = rerank_row["query"]
    ground_truth = rerank_row.get("ground_truth")
    gold_chunk_ids = rerank_row.get("gold_chunk_ids", [])
    gold_parent_ids = rerank_row.get("gold_parent_ids", [])

    reranked_top_k = rerank_row.get("reranked_top_k", [])[:top_k]

    contexts = []

    for item in reranked_top_k:
        chunk_id = item["chunk_id"]
        parent_id = item.get("parent_id")
        title = item.get("title", "")
        chunk_text = item.get("chunk_text", "")

        contexts.append(
            {
                "context_rank": item.get("rank"),
                "chunk_id": chunk_id,
                "parent_id": parent_id,
                "title": title,
                "text": chunk_text,
                "bge_rerank_score": item.get("bge_rerank_score"),
                "old_rank": item.get("old_rank"),
                "is_gold_chunk": chunk_id in set(gold_chunk_ids),
            }
        )

    context_metrics = evaluate_contexts(
        contexts=contexts,
        gold_chunk_ids=gold_chunk_ids,
    )

    prompt = build_prompt(
        query=query,
        contexts=contexts,
    )

    return {
        "query_id": query_id,
        "query": query,
        "ground_truth": ground_truth,
        "gold_chunk_ids": gold_chunk_ids,
        "gold_parent_ids": gold_parent_ids,
        "retriever": "best_hybrid_rrf_plus_bge_rerank",
        "top_k_contexts": top_k,
        "contexts": contexts,
        "context_metrics": context_metrics,
        "prompt": prompt,
    }


def main():
    require_file(GOLDEN_EVAL_PATH)
    require_file(RERANK_RESULTS_PATH)

    golden_rows = read_jsonl(GOLDEN_EVAL_PATH)
    rerank_rows = read_jsonl(RERANK_RESULTS_PATH)
    rerank_metrics = read_json(RERANK_METRICS_PATH)

    print("====== 加载数据 ======")
    print("golden_query_count:", len(golden_rows))
    print("rerank_result_count:", len(rerank_rows))
    print("rerank_recall@10:", rerank_metrics.get("recall@10"))
    print("rerank_mrr@10:", rerank_metrics.get("mrr@10"))

    rag_inputs_by_k = {}

    for top_k in TOP_K_VALUES:
        rag_rows = []

        for row in rerank_rows:
            rag_row = make_rag_input_row(
                rerank_row=row,
                top_k=top_k,
            )
            rag_rows.append(rag_row)

        rag_inputs_by_k[top_k] = rag_rows

    write_jsonl(RAG_INPUTS_TOP5_PATH, rag_inputs_by_k[5])
    write_jsonl(RAG_INPUTS_TOP10_PATH, rag_inputs_by_k[10])

    aggregate_metrics = {
        "method": "final_rag_context_pack",
        "retriever": "best_hybrid_rrf_plus_bge_rerank",
        "query_count": len(rerank_rows),
        "source_rerank_recall@10": rerank_metrics.get("recall@10"),
        "source_rerank_mrr@10": rerank_metrics.get("mrr@10"),
    }

    for top_k in TOP_K_VALUES:
        rows = rag_inputs_by_k[top_k]

        hits = [row["context_metrics"]["hit"] for row in rows]
        recalls = [row["context_metrics"]["recall"] for row in rows]
        precisions = [row["context_metrics"]["precision"] for row in rows]
        mrrs = [row["context_metrics"]["mrr"] for row in rows]
        context_counts = [row["context_metrics"]["context_count"] for row in rows]

        aggregate_metrics[f"context_hit@{top_k}"] = mean(hits) if hits else 0.0
        aggregate_metrics[f"context_recall@{top_k}"] = mean(recalls) if recalls else 0.0
        aggregate_metrics[f"context_precision@{top_k}"] = mean(precisions) if precisions else 0.0
        aggregate_metrics[f"context_mrr@{top_k}"] = mean(mrrs) if mrrs else 0.0
        aggregate_metrics[f"avg_context_count@{top_k}"] = mean(context_counts) if context_counts else 0.0

    with CONTEXT_METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(aggregate_metrics, f, ensure_ascii=False, indent=2)

    report_lines = []
    report_lines.append("# Final RAG Context Pack Report")
    report_lines.append("")
    report_lines.append("## 1. Purpose")
    report_lines.append("")
    report_lines.append("This step converts the final reranked retrieval results into RAG-ready input files.")
    report_lines.append("")
    report_lines.append("## 2. Input")
    report_lines.append("")
    report_lines.append(f"- golden_eval: `{GOLDEN_EVAL_PATH.relative_to(PROJECT_ROOT)}`")
    report_lines.append(f"- rerank_results: `{RERANK_RESULTS_PATH.relative_to(PROJECT_ROOT)}`")
    report_lines.append("")
    report_lines.append("## 3. Output")
    report_lines.append("")
    report_lines.append(f"- rag_inputs_top5: `{RAG_INPUTS_TOP5_PATH.relative_to(PROJECT_ROOT)}`")
    report_lines.append(f"- rag_inputs_top10: `{RAG_INPUTS_TOP10_PATH.relative_to(PROJECT_ROOT)}`")
    report_lines.append(f"- context_metrics: `{CONTEXT_METRICS_PATH.relative_to(PROJECT_ROOT)}`")
    report_lines.append("")
    report_lines.append("## 4. Context metrics")
    report_lines.append("")
    report_lines.append("| Metric | Value |")
    report_lines.append("|---|---:|")

    for top_k in TOP_K_VALUES:
        report_lines.append(f"| Context Hit@{top_k} | {aggregate_metrics[f'context_hit@{top_k}']:.4f} |")
        report_lines.append(f"| Context Recall@{top_k} | {aggregate_metrics[f'context_recall@{top_k}']:.4f} |")
        report_lines.append(f"| Context Precision@{top_k} | {aggregate_metrics[f'context_precision@{top_k}']:.4f} |")
        report_lines.append(f"| Context MRR@{top_k} | {aggregate_metrics[f'context_mrr@{top_k}']:.4f} |")
        report_lines.append(f"| Avg Context Count@{top_k} | {aggregate_metrics[f'avg_context_count@{top_k}']:.2f} |")

    report_lines.append("")
    report_lines.append("## 5. Interpretation")
    report_lines.append("")
    report_lines.append("- Top5 is more concise and usually better for answer generation when context noise matters.")
    report_lines.append("- Top10 has higher evidence recall and is safer for multi-hop questions.")
    report_lines.append("- The next step will generate answers using these packed contexts.")
    report_lines.append("")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("====== Final RAG 输入包构造完成 ======")
    print("rag_inputs_top5:", RAG_INPUTS_TOP5_PATH)
    print("rag_inputs_top10:", RAG_INPUTS_TOP10_PATH)
    print("context_metrics:", CONTEXT_METRICS_PATH)
    print("report:", REPORT_PATH)

    print("====== Context 指标 ======")

    for top_k in TOP_K_VALUES:
        print(f"Context Hit@{top_k}: {aggregate_metrics[f'context_hit@{top_k}']:.4f}")
        print(f"Context Recall@{top_k}: {aggregate_metrics[f'context_recall@{top_k}']:.4f}")
        print(f"Context Precision@{top_k}: {aggregate_metrics[f'context_precision@{top_k}']:.4f}")
        print(f"Context MRR@{top_k}: {aggregate_metrics[f'context_mrr@{top_k}']:.4f}")
        print(f"Avg Context Count@{top_k}: {aggregate_metrics[f'avg_context_count@{top_k}']:.2f}")


if __name__ == "__main__":
    main()