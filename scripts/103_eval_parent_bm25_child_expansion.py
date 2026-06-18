import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "hierarchical_rag"

GOLDEN_EVAL_PATH = DATA_DIR / "golden_eval_50.jsonl"
PARENT_DOCS_PATH = DATA_DIR / "parent_docs.jsonl"
CHILD_CHUNKS_PATH = DATA_DIR / "child_chunks.jsonl"
PARENT_CHILD_MAP_PATH = DATA_DIR / "parent_child_map.json"

OUTPUT_RETRIEVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "retrieval"
OUTPUT_EVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "eval"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "reports"

OUTPUT_RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_RETRIEVAL_DIR / "parent_bm25_child_expansion_results.jsonl"
BAD_CASES_PATH = OUTPUT_RETRIEVAL_DIR / "parent_bm25_child_expansion_bad_cases.jsonl"
METRICS_PATH = OUTPUT_EVAL_DIR / "parent_bm25_child_expansion_metrics.json"
REPORT_PATH = OUTPUT_REPORT_DIR / "parent_bm25_child_expansion_report.md"

TOP_PARENT_K_VALUES = [1, 3, 5, 10]
MAX_PARENT_K = max(TOP_PARENT_K_VALUES)

BM25_K1 = 1.5
BM25_B = 0.75


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


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, docs, id_field: str, text_field: str):
        self.docs = docs
        self.id_field = id_field
        self.text_field = text_field

        self.tokens = []
        for doc in docs:
            title = doc.get("title", "")
            text = doc.get(text_field, "")
            full_text = f"{title} {text}"
            self.tokens.append(tokenize(full_text))

        self.doc_lengths = [len(tokens) for tokens in self.tokens]
        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)

        self.term_freqs = []
        self.doc_freq = Counter()

        for tokens in self.tokens:
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            for term in tf:
                self.doc_freq[term] += 1

        self.num_docs = len(docs)
        self.idf = {}

        for term, df in self.doc_freq.items():
            self.idf[term] = math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))

    def score_doc(self, query_terms, doc_index: int):
        tf = self.term_freqs[doc_index]
        doc_length = self.doc_lengths[doc_index]

        score = 0.0

        for term in query_terms:
            if term not in tf:
                continue

            freq = tf[term]
            idf = self.idf.get(term, 0.0)

            numerator = freq * (BM25_K1 + 1)
            denominator = freq + BM25_K1 * (
                1 - BM25_B + BM25_B * doc_length / max(self.avg_doc_length, 1e-9)
            )

            score += idf * numerator / denominator

        return score

    def search(self, query: str, top_k: int):
        query_terms = tokenize(query)

        scored = []
        for i, doc in enumerate(self.docs):
            score = self.score_doc(query_terms, i)
            scored.append((score, i))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for rank, (score, doc_index) in enumerate(scored[:top_k], start=1):
            doc = self.docs[doc_index]
            results.append(
                {
                    "rank": rank,
                    "score": score,
                    "parent_id": doc["parent_id"],
                    "title": doc.get("title", ""),
                    "parent_text": doc.get("parent_text", ""),
                    "source_item_id": doc.get("source_item_id"),
                    "query_id": doc.get("query_id"),
                }
            )

        return results


def evaluate_ranked_parent_ids(retrieved_ids, gold_ids, k):
    gold_set = set(gold_ids)
    top_k_ids = retrieved_ids[:k]
    top_k_set = set(top_k_ids)

    hit_count = len(gold_set & top_k_set)
    hit_at_k = 1.0 if hit_count > 0 else 0.0
    recall_at_k = hit_count / len(gold_set) if gold_set else 0.0

    first_hit_rank = None
    for rank, item_id in enumerate(top_k_ids, start=1):
        if item_id in gold_set:
            first_hit_rank = rank
            break

    mrr_at_k = 1.0 / first_hit_rank if first_hit_rank is not None else 0.0

    return {
        f"parent_hit@{k}": hit_at_k,
        f"parent_recall@{k}": recall_at_k,
        f"parent_mrr@{k}": mrr_at_k,
        f"parent_hit_count@{k}": hit_count,
        f"parent_first_hit_rank@{k}": first_hit_rank,
    }


def evaluate_expanded_child_ids(expanded_child_ids, gold_child_ids, parent_k):
    gold_set = set(gold_child_ids)
    expanded_set = set(expanded_child_ids)

    hit_count = len(gold_set & expanded_set)
    hit = 1.0 if hit_count > 0 else 0.0
    recall = hit_count / len(gold_set) if gold_set else 0.0

    first_hit_rank = None
    for rank, chunk_id in enumerate(expanded_child_ids, start=1):
        if chunk_id in gold_set:
            first_hit_rank = rank
            break

    mrr = 1.0 / first_hit_rank if first_hit_rank is not None else 0.0

    return {
        f"expanded_child_hit@parent{parent_k}": hit,
        f"expanded_child_recall@parent{parent_k}": recall,
        f"expanded_child_mrr@parent{parent_k}": mrr,
        f"expanded_child_hit_count@parent{parent_k}": hit_count,
        f"expanded_child_candidate_count@parent{parent_k}": len(expanded_child_ids),
    }


def main():
    if not GOLDEN_EVAL_PATH.exists():
        raise FileNotFoundError(f"Missing file: {GOLDEN_EVAL_PATH}")

    if not PARENT_DOCS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {PARENT_DOCS_PATH}")

    if not CHILD_CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {CHILD_CHUNKS_PATH}")

    if not PARENT_CHILD_MAP_PATH.exists():
        raise FileNotFoundError(f"Missing file: {PARENT_CHILD_MAP_PATH}")

    golden_rows = read_jsonl(GOLDEN_EVAL_PATH)
    parent_docs = read_jsonl(PARENT_DOCS_PATH)
    child_chunks = read_jsonl(CHILD_CHUNKS_PATH)

    with PARENT_CHILD_MAP_PATH.open("r", encoding="utf-8") as f:
        parent_child_map = json.load(f)

    chunk_by_id = {row["chunk_id"]: row for row in child_chunks}

    print("====== 加载数据 ======")
    print("golden_query_count:", len(golden_rows))
    print("parent_doc_count:", len(parent_docs))
    print("child_chunk_count:", len(child_chunks))

    print("====== 构建 parent-level BM25 index ======")
    parent_index = BM25Index(
        docs=parent_docs,
        id_field="parent_id",
        text_field="parent_text",
    )

    result_rows = []
    bad_cases = []

    metric_lists = {}

    for k in TOP_PARENT_K_VALUES:
        metric_lists[f"parent_hit@{k}"] = []
        metric_lists[f"parent_recall@{k}"] = []
        metric_lists[f"parent_mrr@{k}"] = []

        metric_lists[f"expanded_child_hit@parent{k}"] = []
        metric_lists[f"expanded_child_recall@parent{k}"] = []
        metric_lists[f"expanded_child_mrr@parent{k}"] = []
        metric_lists[f"expanded_child_candidate_count@parent{k}"] = []

    for row in golden_rows:
        query_id = row["query_id"]
        query = row["query"]
        gold_parent_ids = row.get("gold_parent_ids", [])
        gold_chunk_ids = row.get("gold_chunk_ids", [])

        retrieved_parents = parent_index.search(query, top_k=MAX_PARENT_K)
        retrieved_parent_ids = [item["parent_id"] for item in retrieved_parents]

        per_query_metrics = {}
        expansion_by_parent_k = {}

        for parent_k in TOP_PARENT_K_VALUES:
            parent_metrics = evaluate_ranked_parent_ids(
                retrieved_ids=retrieved_parent_ids,
                gold_ids=gold_parent_ids,
                k=parent_k,
            )

            per_query_metrics.update(parent_metrics)

            metric_lists[f"parent_hit@{parent_k}"].append(parent_metrics[f"parent_hit@{parent_k}"])
            metric_lists[f"parent_recall@{parent_k}"].append(parent_metrics[f"parent_recall@{parent_k}"])
            metric_lists[f"parent_mrr@{parent_k}"].append(parent_metrics[f"parent_mrr@{parent_k}"])

            top_parent_ids = retrieved_parent_ids[:parent_k]

            expanded_child_ids = []
            expanded_child_rows = []

            for parent_id in top_parent_ids:
                child_ids = parent_child_map.get(parent_id, [])

                for child_id in child_ids:
                    if child_id not in expanded_child_ids:
                        expanded_child_ids.append(child_id)

                        if child_id in chunk_by_id:
                            expanded_child_rows.append(chunk_by_id[child_id])

            child_metrics = evaluate_expanded_child_ids(
                expanded_child_ids=expanded_child_ids,
                gold_child_ids=gold_chunk_ids,
                parent_k=parent_k,
            )

            per_query_metrics.update(child_metrics)

            metric_lists[f"expanded_child_hit@parent{parent_k}"].append(
                child_metrics[f"expanded_child_hit@parent{parent_k}"]
            )
            metric_lists[f"expanded_child_recall@parent{parent_k}"].append(
                child_metrics[f"expanded_child_recall@parent{parent_k}"]
            )
            metric_lists[f"expanded_child_mrr@parent{parent_k}"].append(
                child_metrics[f"expanded_child_mrr@parent{parent_k}"]
            )
            metric_lists[f"expanded_child_candidate_count@parent{parent_k}"].append(
                child_metrics[f"expanded_child_candidate_count@parent{parent_k}"]
            )

            expansion_by_parent_k[f"parent_top_{parent_k}"] = {
                "expanded_child_count": len(expanded_child_ids),
                "expanded_child_ids": expanded_child_ids,
                "expanded_child_preview": expanded_child_rows[:5],
            }

        result_row = {
            "query_id": query_id,
            "query": query,
            "ground_truth": row.get("ground_truth"),
            "question_type": row.get("question_type"),
            "difficulty": row.get("difficulty"),
            "gold_parent_ids": gold_parent_ids,
            "gold_chunk_ids": gold_chunk_ids,
            "retrieved_parent_ids": retrieved_parent_ids,
            "retrieved_parents": retrieved_parents,
            "expansion_by_parent_k": expansion_by_parent_k,
            "metrics": per_query_metrics,
        }

        result_rows.append(result_row)

        if per_query_metrics.get("expanded_child_hit@parent10", 0.0) == 0.0:
            bad_cases.append(result_row)

    aggregate_metrics = {
        "method": "parent_bm25_child_expansion",
        "golden_query_count": len(golden_rows),
        "parent_doc_count": len(parent_docs),
        "child_chunk_count": len(child_chunks),
        "top_parent_k_values": TOP_PARENT_K_VALUES,
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
    }

    for metric_name, values in metric_lists.items():
        aggregate_metrics[metric_name] = mean(values) if values else 0.0

    write_jsonl(RESULTS_PATH, result_rows)
    write_jsonl(BAD_CASES_PATH, bad_cases)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(aggregate_metrics, f, ensure_ascii=False, indent=2)

    report_lines = []
    report_lines.append("# Parent BM25 + Child Expansion Report")
    report_lines.append("")
    report_lines.append("## 1. Method")
    report_lines.append("")
    report_lines.append("- method: `BM25 over parent_docs + child expansion`")
    report_lines.append(f"- bm25_k1: `{BM25_K1}`")
    report_lines.append(f"- bm25_b: `{BM25_B}`")
    report_lines.append("")
    report_lines.append("## 2. Data")
    report_lines.append("")
    report_lines.append(f"- golden_query_count: `{len(golden_rows)}`")
    report_lines.append(f"- parent_doc_count: `{len(parent_docs)}`")
    report_lines.append(f"- child_chunk_count: `{len(child_chunks)}`")
    report_lines.append("")
    report_lines.append("## 3. Parent retrieval metrics")
    report_lines.append("")
    report_lines.append("| Metric | Value |")
    report_lines.append("|---|---:|")

    for k in TOP_PARENT_K_VALUES:
        report_lines.append(f"| Parent Hit@{k} | {aggregate_metrics[f'parent_hit@{k}']:.4f} |")
        report_lines.append(f"| Parent Recall@{k} | {aggregate_metrics[f'parent_recall@{k}']:.4f} |")
        report_lines.append(f"| Parent MRR@{k} | {aggregate_metrics[f'parent_mrr@{k}']:.4f} |")

    report_lines.append("")
    report_lines.append("## 4. Expanded child coverage")
    report_lines.append("")
    report_lines.append("| Metric | Value |")
    report_lines.append("|---|---:|")

    for k in TOP_PARENT_K_VALUES:
        report_lines.append(f"| Expanded Child Hit@Parent{k} | {aggregate_metrics[f'expanded_child_hit@parent{k}']:.4f} |")
        report_lines.append(f"| Expanded Child Recall@Parent{k} | {aggregate_metrics[f'expanded_child_recall@parent{k}']:.4f} |")
        report_lines.append(f"| Expanded Child MRR@Parent{k} | {aggregate_metrics[f'expanded_child_mrr@parent{k}']:.4f} |")
        report_lines.append(f"| Avg Candidate Count@Parent{k} | {aggregate_metrics[f'expanded_child_candidate_count@parent{k}']:.2f} |")

    report_lines.append("")
    report_lines.append("## 5. Bad cases")
    report_lines.append("")
    report_lines.append(f"- no_expanded_child_hit_at_parent10_count: `{len(bad_cases)}`")
    report_lines.append(f"- bad_cases_file: `{BAD_CASES_PATH.relative_to(PROJECT_ROOT)}`")
    report_lines.append("")
    report_lines.append("## 6. Interpretation")
    report_lines.append("")
    report_lines.append("- Parent retrieval checks whether the system can find relevant parent documents.")
    report_lines.append("- Child expansion checks whether the child chunks under retrieved parents can cover gold evidence.")
    report_lines.append("- High expanded child recall with high candidate count means parent retrieval is useful, but reranking is still needed.")
    report_lines.append("")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("====== Parent BM25 + Child Expansion 完成 ======")
    print("results:", RESULTS_PATH)
    print("bad_cases:", BAD_CASES_PATH)
    print("metrics:", METRICS_PATH)
    print("report:", REPORT_PATH)

    print("====== Parent 检索指标 ======")
    for k in TOP_PARENT_K_VALUES:
        print(f"Parent Hit@{k}: {aggregate_metrics[f'parent_hit@{k}']:.4f}")
        print(f"Parent Recall@{k}: {aggregate_metrics[f'parent_recall@{k}']:.4f}")
        print(f"Parent MRR@{k}: {aggregate_metrics[f'parent_mrr@{k}']:.4f}")

    print("====== Child Expansion 覆盖指标 ======")
    for k in TOP_PARENT_K_VALUES:
        print(f"Expanded Child Hit@Parent{k}: {aggregate_metrics[f'expanded_child_hit@parent{k}']:.4f}")
        print(f"Expanded Child Recall@Parent{k}: {aggregate_metrics[f'expanded_child_recall@parent{k}']:.4f}")
        print(f"Expanded Child MRR@Parent{k}: {aggregate_metrics[f'expanded_child_mrr@parent{k}']:.4f}")
        print(f"Avg Candidate Count@Parent{k}: {aggregate_metrics[f'expanded_child_candidate_count@parent{k}']:.2f}")


if __name__ == "__main__":
    main()