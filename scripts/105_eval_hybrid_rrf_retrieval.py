import json
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "hierarchical_rag"
RETRIEVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "retrieval"
EVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "eval"
REPORT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "reports"

GOLDEN_EVAL_PATH = DATA_DIR / "golden_eval_50.jsonl"
CHILD_CHUNKS_PATH = DATA_DIR / "child_chunks.jsonl"

BM25_CHILD_RESULTS_PATH = RETRIEVAL_DIR / "bm25_child_retrieval_results.jsonl"
EMBEDDING_CHILD_RESULTS_PATH = RETRIEVAL_DIR / "embedding_child_retrieval_results.jsonl"
PARENT_EXPANSION_RESULTS_PATH = RETRIEVAL_DIR / "parent_bm25_child_expansion_results.jsonl"

BM25_CHILD_METRICS_PATH = EVAL_DIR / "bm25_child_metrics.json"
EMBEDDING_CHILD_METRICS_PATH = EVAL_DIR / "embedding_child_metrics.json"
PARENT_EXPANSION_METRICS_PATH = EVAL_DIR / "parent_bm25_child_expansion_metrics.json"

RESULTS_PATH = RETRIEVAL_DIR / "hybrid_rrf_retrieval_results.jsonl"
BAD_CASES_PATH = RETRIEVAL_DIR / "hybrid_rrf_bad_cases.jsonl"
METRICS_PATH = EVAL_DIR / "hybrid_rrf_metrics.json"
REPORT_PATH = REPORT_DIR / "hybrid_rrf_retrieval_report.md"

RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K_VALUES = [1, 3, 5, 10]
MAX_RETURN_K = max(TOP_K_VALUES)

RRF_K = 60

SOURCE_WEIGHTS = {
    "bm25_child": 1.0,
    "embedding_child": 1.0,
    "parent_expansion": 0.8,
}


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


def read_json_if_exists(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")


def evaluate_at_k(retrieved_ids, gold_ids, k):
    gold_set = set(gold_ids)
    top_k_ids = retrieved_ids[:k]
    top_k_set = set(top_k_ids)

    hit_count = len(gold_set & top_k_set)

    hit_at_k = 1.0 if hit_count > 0 else 0.0
    recall_at_k = hit_count / len(gold_set) if gold_set else 0.0

    first_hit_rank = None
    for rank, chunk_id in enumerate(top_k_ids, start=1):
        if chunk_id in gold_set:
            first_hit_rank = rank
            break

    mrr_at_k = 1.0 / first_hit_rank if first_hit_rank is not None else 0.0

    return {
        f"hit@{k}": hit_at_k,
        f"recall@{k}": recall_at_k,
        f"mrr@{k}": mrr_at_k,
        f"hit_count@{k}": hit_count,
        f"first_hit_rank@{k}": first_hit_rank,
    }


def add_rrf_scores(score_table, chunk_ids, source_name, weight):
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        rrf_score = weight * (1.0 / (RRF_K + rank))

        if chunk_id not in score_table:
            score_table[chunk_id] = {
                "chunk_id": chunk_id,
                "rrf_score": 0.0,
                "source_hits": [],
            }

        score_table[chunk_id]["rrf_score"] += rrf_score
        score_table[chunk_id]["source_hits"].append(
            {
                "source": source_name,
                "rank": rank,
                "weight": weight,
                "rrf_score": rrf_score,
            }
        )


def get_parent_expanded_ids(parent_result_row):
    expansion = parent_result_row.get("expansion_by_parent_k", {})

    parent_top_10 = expansion.get("parent_top_10", {})
    expanded_ids = parent_top_10.get("expanded_child_ids", [])

    if expanded_ids:
        return expanded_ids

    parent_top_5 = expansion.get("parent_top_5", {})
    expanded_ids = parent_top_5.get("expanded_child_ids", [])

    return expanded_ids


def main():
    for path in [
        GOLDEN_EVAL_PATH,
        CHILD_CHUNKS_PATH,
        BM25_CHILD_RESULTS_PATH,
        EMBEDDING_CHILD_RESULTS_PATH,
        PARENT_EXPANSION_RESULTS_PATH,
    ]:
        require_file(path)

    golden_rows = read_jsonl(GOLDEN_EVAL_PATH)
    child_chunks = read_jsonl(CHILD_CHUNKS_PATH)

    bm25_rows = read_jsonl(BM25_CHILD_RESULTS_PATH)
    embedding_rows = read_jsonl(EMBEDDING_CHILD_RESULTS_PATH)
    parent_rows = read_jsonl(PARENT_EXPANSION_RESULTS_PATH)

    child_by_id = {row["chunk_id"]: row for row in child_chunks}

    bm25_by_qid = {row["query_id"]: row for row in bm25_rows}
    embedding_by_qid = {row["query_id"]: row for row in embedding_rows}
    parent_by_qid = {row["query_id"]: row for row in parent_rows}

    print("====== 加载数据 ======")
    print("golden_query_count:", len(golden_rows))
    print("child_chunk_count:", len(child_chunks))
    print("bm25_result_count:", len(bm25_rows))
    print("embedding_result_count:", len(embedding_rows))
    print("parent_expansion_result_count:", len(parent_rows))

    result_rows = []
    bad_cases = []

    metric_lists = {}
    for k in TOP_K_VALUES:
        metric_lists[f"hit@{k}"] = []
        metric_lists[f"recall@{k}"] = []
        metric_lists[f"mrr@{k}"] = []

    print("====== 开始 Hybrid RRF 融合评估 ======")

    for index, golden_row in enumerate(golden_rows, start=1):
        query_id = golden_row["query_id"]
        query = golden_row["query"]
        gold_chunk_ids = golden_row.get("gold_chunk_ids", [])

        if query_id not in bm25_by_qid:
            raise KeyError(f"BM25 child results 缺少 query_id: {query_id}")

        if query_id not in embedding_by_qid:
            raise KeyError(f"Embedding child results 缺少 query_id: {query_id}")

        if query_id not in parent_by_qid:
            raise KeyError(f"Parent expansion results 缺少 query_id: {query_id}")

        bm25_ids = bm25_by_qid[query_id].get("retrieved_chunk_ids", [])
        embedding_ids = embedding_by_qid[query_id].get("retrieved_chunk_ids", [])
        parent_expanded_ids = get_parent_expanded_ids(parent_by_qid[query_id])

        score_table = {}

        add_rrf_scores(
            score_table=score_table,
            chunk_ids=bm25_ids,
            source_name="bm25_child",
            weight=SOURCE_WEIGHTS["bm25_child"],
        )

        add_rrf_scores(
            score_table=score_table,
            chunk_ids=embedding_ids,
            source_name="embedding_child",
            weight=SOURCE_WEIGHTS["embedding_child"],
        )

        add_rrf_scores(
            score_table=score_table,
            chunk_ids=parent_expanded_ids,
            source_name="parent_expansion",
            weight=SOURCE_WEIGHTS["parent_expansion"],
        )

        ranked_items = sorted(
            score_table.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        retrieved = []
        for rank, item in enumerate(ranked_items[:MAX_RETURN_K], start=1):
            chunk_id = item["chunk_id"]
            chunk = child_by_id.get(chunk_id, {})

            retrieved.append(
                {
                    "rank": rank,
                    "rrf_score": item["rrf_score"],
                    "chunk_id": chunk_id,
                    "parent_id": chunk.get("parent_id"),
                    "title": chunk.get("title", ""),
                    "chunk_text": chunk.get("chunk_text", ""),
                    "source_hits": item["source_hits"],
                }
            )

        retrieved_ids = [item["chunk_id"] for item in retrieved]

        per_query_metrics = {}

        for k in TOP_K_VALUES:
            metrics = evaluate_at_k(
                retrieved_ids=retrieved_ids,
                gold_ids=gold_chunk_ids,
                k=k,
            )

            per_query_metrics.update(metrics)

            metric_lists[f"hit@{k}"].append(metrics[f"hit@{k}"])
            metric_lists[f"recall@{k}"].append(metrics[f"recall@{k}"])
            metric_lists[f"mrr@{k}"].append(metrics[f"mrr@{k}"])

        result_row = {
            "query_id": query_id,
            "query": query,
            "ground_truth": golden_row.get("ground_truth"),
            "question_type": golden_row.get("question_type"),
            "difficulty": golden_row.get("difficulty"),
            "gold_parent_ids": golden_row.get("gold_parent_ids", []),
            "gold_chunk_ids": gold_chunk_ids,
            "bm25_child_ids": bm25_ids,
            "embedding_child_ids": embedding_ids,
            "parent_expanded_child_ids": parent_expanded_ids,
            "hybrid_retrieved_chunk_ids": retrieved_ids,
            "hybrid_retrieved_top_k": retrieved,
            "metrics": per_query_metrics,
        }

        result_rows.append(result_row)

        if per_query_metrics.get("hit@10", 0.0) == 0.0:
            bad_cases.append(result_row)

        print(f"evaluated {index}/{len(golden_rows)}")

    aggregate_metrics = {
        "method": "hybrid_rrf",
        "golden_query_count": len(golden_rows),
        "child_chunk_count": len(child_chunks),
        "top_k_values": TOP_K_VALUES,
        "rrf_k": RRF_K,
        "source_weights": SOURCE_WEIGHTS,
        "sources": [
            "bm25_child",
            "embedding_child",
            "parent_bm25_child_expansion",
        ],
    }

    for metric_name, values in metric_lists.items():
        aggregate_metrics[metric_name] = mean(values) if values else 0.0

    previous_metrics = {
        "bm25_child": read_json_if_exists(BM25_CHILD_METRICS_PATH),
        "embedding_child": read_json_if_exists(EMBEDDING_CHILD_METRICS_PATH),
        "parent_expansion": read_json_if_exists(PARENT_EXPANSION_METRICS_PATH),
    }

    aggregate_metrics["previous_metrics_summary"] = {
        "bm25_child_recall@10": previous_metrics["bm25_child"].get("recall@10"),
        "embedding_child_recall@10": previous_metrics["embedding_child"].get("recall@10"),
        "parent_expanded_child_recall@parent10": previous_metrics["parent_expansion"].get(
            "expanded_child_recall@parent10"
        ),
    }

    write_jsonl(RESULTS_PATH, result_rows)
    write_jsonl(BAD_CASES_PATH, bad_cases)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(aggregate_metrics, f, ensure_ascii=False, indent=2)

    report_lines = []
    report_lines.append("# Hybrid RRF Retrieval Report")
    report_lines.append("")
    report_lines.append("## 1. Method")
    report_lines.append("")
    report_lines.append("- method: `Reciprocal Rank Fusion`")
    report_lines.append(f"- rrf_k: `{RRF_K}`")
    report_lines.append(f"- source_weights: `{SOURCE_WEIGHTS}`")
    report_lines.append("")
    report_lines.append("## 2. Sources")
    report_lines.append("")
    report_lines.append("- BM25 child retrieval")
    report_lines.append("- Embedding child retrieval")
    report_lines.append("- Parent BM25 + child expansion")
    report_lines.append("")
    report_lines.append("## 3. Current Hybrid RRF metrics")
    report_lines.append("")
    report_lines.append("| Metric | Value |")
    report_lines.append("|---|---:|")

    for k in TOP_K_VALUES:
        report_lines.append(f"| Hit@{k} | {aggregate_metrics[f'hit@{k}']:.4f} |")
        report_lines.append(f"| Recall@{k} | {aggregate_metrics[f'recall@{k}']:.4f} |")
        report_lines.append(f"| MRR@{k} | {aggregate_metrics[f'mrr@{k}']:.4f} |")

    report_lines.append("")
    report_lines.append("## 4. Previous baseline comparison")
    report_lines.append("")
    report_lines.append("| Method | Recall@10 |")
    report_lines.append("|---|---:|")
    report_lines.append(
        f"| BM25 child | {previous_metrics['bm25_child'].get('recall@10', 0.0):.4f} |"
    )
    report_lines.append(
        f"| Embedding child | {previous_metrics['embedding_child'].get('recall@10', 0.0):.4f} |"
    )
    report_lines.append(
        f"| Parent expansion | {previous_metrics['parent_expansion'].get('expanded_child_recall@parent10', 0.0):.4f} |"
    )
    report_lines.append(
        f"| Hybrid RRF | {aggregate_metrics['recall@10']:.4f} |"
    )
    report_lines.append("")
    report_lines.append("## 5. Interpretation")
    report_lines.append("")
    report_lines.append("- Hybrid RRF combines sparse lexical matching, dense semantic matching, and parent-level expansion.")
    report_lines.append("- If Hybrid RRF improves Recall@10, it becomes the best retriever candidate before reranking.")
    report_lines.append("- If it does not improve over embedding alone, the project can still report that BGE embedding is already strong and RRF adds limited value on this 50-query benchmark.")
    report_lines.append("")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("====== Hybrid RRF retrieval 完成 ======")
    print("results:", RESULTS_PATH)
    print("bad_cases:", BAD_CASES_PATH)
    print("metrics:", METRICS_PATH)
    print("report:", REPORT_PATH)

    print("====== 核心指标 ======")
    for k in TOP_K_VALUES:
        print(f"Hit@{k}: {aggregate_metrics[f'hit@{k}']:.4f}")
        print(f"Recall@{k}: {aggregate_metrics[f'recall@{k}']:.4f}")
        print(f"MRR@{k}: {aggregate_metrics[f'mrr@{k}']:.4f}")

    print("====== Recall@10 对比 ======")
    print("BM25 child Recall@10:", previous_metrics["bm25_child"].get("recall@10"))
    print("Embedding child Recall@10:", previous_metrics["embedding_child"].get("recall@10"))
    print(
        "Parent expansion Recall@Parent10:",
        previous_metrics["parent_expansion"].get("expanded_child_recall@parent10"),
    )
    print("Hybrid RRF Recall@10:", aggregate_metrics["recall@10"])


if __name__ == "__main__":
    main()