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

OUTPUT_GRID_PATH = EVAL_DIR / "hybrid_rrf_grid_search_results.json"
OUTPUT_BEST_RESULTS_PATH = RETRIEVAL_DIR / "hybrid_rrf_best_retrieval_results.jsonl"
OUTPUT_BEST_BAD_CASES_PATH = RETRIEVAL_DIR / "hybrid_rrf_best_bad_cases.jsonl"
OUTPUT_BEST_METRICS_PATH = EVAL_DIR / "hybrid_rrf_best_metrics.json"
REPORT_PATH = REPORT_DIR / "hybrid_rrf_grid_search_report.md"

TOP_K_VALUES = [1, 3, 5, 10]
MAX_RETURN_K = max(TOP_K_VALUES)

RRF_K_VALUES = [10, 30, 60, 100]

BM25_WEIGHTS = [0.0, 0.25, 0.5, 1.0]
EMBEDDING_WEIGHTS = [1.0, 2.0, 3.0, 5.0]
PARENT_WEIGHTS = [0.0, 0.25, 0.5, 0.8, 1.0]


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
    }


def get_parent_expanded_ids(parent_result_row):
    expansion = parent_result_row.get("expansion_by_parent_k", {})

    for key in ["parent_top_10", "parent_top_5", "parent_top_3", "parent_top_1"]:
        item = expansion.get(key, {})
        expanded_ids = item.get("expanded_child_ids", [])
        if expanded_ids:
            return expanded_ids

    return []


def add_rrf_scores(score_table, chunk_ids, source_name, weight, rrf_k):
    if weight <= 0:
        return

    for rank, chunk_id in enumerate(chunk_ids, start=1):
        score = weight * (1.0 / (rrf_k + rank))

        if chunk_id not in score_table:
            score_table[chunk_id] = {
                "chunk_id": chunk_id,
                "rrf_score": 0.0,
                "source_hits": [],
            }

        score_table[chunk_id]["rrf_score"] += score
        score_table[chunk_id]["source_hits"].append(
            {
                "source": source_name,
                "rank": rank,
                "weight": weight,
                "rrf_score": score,
            }
        )


def run_one_config(
    golden_rows,
    child_by_id,
    bm25_by_qid,
    embedding_by_qid,
    parent_by_qid,
    rrf_k,
    bm25_weight,
    embedding_weight,
    parent_weight,
    keep_details=False,
):
    result_rows = []
    bad_cases = []

    metric_lists = {}
    for k in TOP_K_VALUES:
        metric_lists[f"hit@{k}"] = []
        metric_lists[f"recall@{k}"] = []
        metric_lists[f"mrr@{k}"] = []

    for golden_row in golden_rows:
        query_id = golden_row["query_id"]
        gold_chunk_ids = golden_row.get("gold_chunk_ids", [])

        bm25_ids = bm25_by_qid[query_id].get("retrieved_chunk_ids", [])
        embedding_ids = embedding_by_qid[query_id].get("retrieved_chunk_ids", [])
        parent_ids = get_parent_expanded_ids(parent_by_qid[query_id])

        score_table = {}

        add_rrf_scores(
            score_table=score_table,
            chunk_ids=bm25_ids,
            source_name="bm25_child",
            weight=bm25_weight,
            rrf_k=rrf_k,
        )

        add_rrf_scores(
            score_table=score_table,
            chunk_ids=embedding_ids,
            source_name="embedding_child",
            weight=embedding_weight,
            rrf_k=rrf_k,
        )

        add_rrf_scores(
            score_table=score_table,
            chunk_ids=parent_ids,
            source_name="parent_expansion",
            weight=parent_weight,
            rrf_k=rrf_k,
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

        if keep_details:
            result_row = {
                "query_id": query_id,
                "query": golden_row.get("query"),
                "ground_truth": golden_row.get("ground_truth"),
                "question_type": golden_row.get("question_type"),
                "difficulty": golden_row.get("difficulty"),
                "gold_parent_ids": golden_row.get("gold_parent_ids", []),
                "gold_chunk_ids": gold_chunk_ids,
                "bm25_child_ids": bm25_ids,
                "embedding_child_ids": embedding_ids,
                "parent_expanded_child_ids": parent_ids,
                "hybrid_retrieved_chunk_ids": retrieved_ids,
                "hybrid_retrieved_top_k": retrieved,
                "metrics": per_query_metrics,
            }

            result_rows.append(result_row)

            if per_query_metrics.get("hit@10", 0.0) == 0.0:
                bad_cases.append(result_row)

    aggregate = {
        "method": "hybrid_rrf_grid_search",
        "rrf_k": rrf_k,
        "bm25_weight": bm25_weight,
        "embedding_weight": embedding_weight,
        "parent_weight": parent_weight,
    }

    for metric_name, values in metric_lists.items():
        aggregate[metric_name] = mean(values) if values else 0.0

    return aggregate, result_rows, bad_cases


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

    all_configs = []

    total = (
        len(RRF_K_VALUES)
        * len(BM25_WEIGHTS)
        * len(EMBEDDING_WEIGHTS)
        * len(PARENT_WEIGHTS)
    )

    current = 0

    print("====== 开始 RRF 权重搜索 ======")

    for rrf_k in RRF_K_VALUES:
        for bm25_weight in BM25_WEIGHTS:
            for embedding_weight in EMBEDDING_WEIGHTS:
                for parent_weight in PARENT_WEIGHTS:
                    current += 1

                    aggregate, _, _ = run_one_config(
                        golden_rows=golden_rows,
                        child_by_id=child_by_id,
                        bm25_by_qid=bm25_by_qid,
                        embedding_by_qid=embedding_by_qid,
                        parent_by_qid=parent_by_qid,
                        rrf_k=rrf_k,
                        bm25_weight=bm25_weight,
                        embedding_weight=embedding_weight,
                        parent_weight=parent_weight,
                        keep_details=False,
                    )

                    all_configs.append(aggregate)

                    print(
                        f"searched {current}/{total} | "
                        f"rrf_k={rrf_k}, "
                        f"bm25={bm25_weight}, "
                        f"emb={embedding_weight}, "
                        f"parent={parent_weight}, "
                        f"recall@10={aggregate['recall@10']:.4f}",
                        flush=True,
                    )

    all_configs.sort(
        key=lambda item: (
            item["recall@10"],
            item["hit@10"],
            item["mrr@10"],
            item["recall@5"],
            item["mrr@5"],
        ),
        reverse=True,
    )

    best_config = all_configs[0]

    print("====== 最优配置 ======")
    print(json.dumps(best_config, ensure_ascii=False, indent=2))

    best_metrics, best_rows, best_bad_cases = run_one_config(
        golden_rows=golden_rows,
        child_by_id=child_by_id,
        bm25_by_qid=bm25_by_qid,
        embedding_by_qid=embedding_by_qid,
        parent_by_qid=parent_by_qid,
        rrf_k=best_config["rrf_k"],
        bm25_weight=best_config["bm25_weight"],
        embedding_weight=best_config["embedding_weight"],
        parent_weight=best_config["parent_weight"],
        keep_details=True,
    )

    output = {
        "searched_config_count": len(all_configs),
        "top_20_configs": all_configs[:20],
        "best_config": best_config,
    }

    with OUTPUT_GRID_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    write_jsonl(OUTPUT_BEST_RESULTS_PATH, best_rows)
    write_jsonl(OUTPUT_BEST_BAD_CASES_PATH, best_bad_cases)

    with OUTPUT_BEST_METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, ensure_ascii=False, indent=2)

    report_lines = []
    report_lines.append("# Hybrid RRF Grid Search Report")
    report_lines.append("")
    report_lines.append("## 1. Purpose")
    report_lines.append("")
    report_lines.append("This experiment searches RRF weights because the default Hybrid RRF configuration underperformed the embedding-only retriever.")
    report_lines.append("")
    report_lines.append("## 2. Search space")
    report_lines.append("")
    report_lines.append(f"- RRF_K_VALUES: `{RRF_K_VALUES}`")
    report_lines.append(f"- BM25_WEIGHTS: `{BM25_WEIGHTS}`")
    report_lines.append(f"- EMBEDDING_WEIGHTS: `{EMBEDDING_WEIGHTS}`")
    report_lines.append(f"- PARENT_WEIGHTS: `{PARENT_WEIGHTS}`")
    report_lines.append(f"- searched_config_count: `{len(all_configs)}`")
    report_lines.append("")
    report_lines.append("## 3. Best config")
    report_lines.append("")
    report_lines.append(f"- rrf_k: `{best_config['rrf_k']}`")
    report_lines.append(f"- bm25_weight: `{best_config['bm25_weight']}`")
    report_lines.append(f"- embedding_weight: `{best_config['embedding_weight']}`")
    report_lines.append(f"- parent_weight: `{best_config['parent_weight']}`")
    report_lines.append("")
    report_lines.append("## 4. Best metrics")
    report_lines.append("")
    report_lines.append("| Metric | Value |")
    report_lines.append("|---|---:|")

    for k in TOP_K_VALUES:
        report_lines.append(f"| Hit@{k} | {best_config[f'hit@{k}']:.4f} |")
        report_lines.append(f"| Recall@{k} | {best_config[f'recall@{k}']:.4f} |")
        report_lines.append(f"| MRR@{k} | {best_config[f'mrr@{k}']:.4f} |")

    report_lines.append("")
    report_lines.append("## 5. Top 10 configs")
    report_lines.append("")
    report_lines.append("| Rank | rrf_k | bm25 | embedding | parent | Recall@10 | Hit@10 | MRR@10 |")
    report_lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")

    for rank, item in enumerate(all_configs[:10], start=1):
        report_lines.append(
            f"| {rank} | {item['rrf_k']} | {item['bm25_weight']} | "
            f"{item['embedding_weight']} | {item['parent_weight']} | "
            f"{item['recall@10']:.4f} | {item['hit@10']:.4f} | {item['mrr@10']:.4f} |"
        )

    report_lines.append("")
    report_lines.append("## 6. Interpretation")
    report_lines.append("")
    report_lines.append("- If the best config is embedding-heavy, it means dense retrieval is the strongest signal on this benchmark.")
    report_lines.append("- If BM25 or parent weights hurt Recall@10, they should be treated as ablation components rather than the final retriever.")
    report_lines.append("- This gives a defensible experiment instead of blindly claiming hybrid retrieval is always better.")
    report_lines.append("")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("====== RRF Grid Search 完成 ======")
    print("grid_results:", OUTPUT_GRID_PATH)
    print("best_results:", OUTPUT_BEST_RESULTS_PATH)
    print("best_bad_cases:", OUTPUT_BEST_BAD_CASES_PATH)
    print("best_metrics:", OUTPUT_BEST_METRICS_PATH)
    print("report:", REPORT_PATH)

    print("====== 最优核心指标 ======")
    print("rrf_k:", best_config["rrf_k"])
    print("bm25_weight:", best_config["bm25_weight"])
    print("embedding_weight:", best_config["embedding_weight"])
    print("parent_weight:", best_config["parent_weight"])

    for k in TOP_K_VALUES:
        print(f"Hit@{k}: {best_config[f'hit@{k}']:.4f}")
        print(f"Recall@{k}: {best_config[f'recall@{k}']:.4f}")
        print(f"MRR@{k}: {best_config[f'mrr@{k}']:.4f}")


if __name__ == "__main__":
    main()