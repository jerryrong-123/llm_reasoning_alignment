import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "hierarchical_rag"
GOLDEN_EVAL_PATH = DATA_DIR / "golden_eval_50.jsonl"
CHILD_CHUNKS_PATH = DATA_DIR / "child_chunks.jsonl"

OUTPUT_RETRIEVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "retrieval"
OUTPUT_EVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "eval"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "reports"

OUTPUT_RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_RETRIEVAL_DIR / "bm25_child_retrieval_results.jsonl"
BAD_CASES_PATH = OUTPUT_RETRIEVAL_DIR / "bm25_child_bad_cases.jsonl"
METRICS_PATH = OUTPUT_EVAL_DIR / "bm25_child_metrics.json"
REPORT_PATH = OUTPUT_REPORT_DIR / "bm25_child_retrieval_report.md"

TOP_K_VALUES = [1, 3, 5, 10]
MAX_RETURN_K = max(TOP_K_VALUES)

BM25_K1 = 1.5
BM25_B = 0.75


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, docs):
        self.docs = docs
        self.tokens = [tokenize(doc.get("chunk_text", "")) for doc in docs]
        self.doc_lengths = [len(tokens) for tokens in self.tokens]
        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)

        self.term_freqs = []
        self.doc_freq = Counter()

        for tokens in self.tokens:
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            for term in tf.keys():
                self.doc_freq[term] += 1

        self.num_docs = len(docs)
        self.idf = {}

        for term, df in self.doc_freq.items():
            self.idf[term] = math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))

    def score_one_doc(self, query_terms, doc_index):
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

    def search(self, query, top_k):
        query_terms = tokenize(query)

        scored = []
        for i, doc in enumerate(self.docs):
            score = self.score_one_doc(query_terms, i)
            scored.append((score, i))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for rank, (score, doc_index) in enumerate(scored[:top_k], start=1):
            doc = self.docs[doc_index]
            results.append(
                {
                    "rank": rank,
                    "score": score,
                    "chunk_id": doc["chunk_id"],
                    "parent_id": doc["parent_id"],
                    "title": doc.get("title", ""),
                    "chunk_text": doc.get("chunk_text", ""),
                    "start_sentence": doc.get("start_sentence"),
                    "end_sentence_exclusive": doc.get("end_sentence_exclusive"),
                    "source_item_id": doc.get("source_item_id"),
                }
            )

        return results


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


def main():
    if not GOLDEN_EVAL_PATH.exists():
        raise FileNotFoundError(f"Missing file: {GOLDEN_EVAL_PATH}")

    if not CHILD_CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {CHILD_CHUNKS_PATH}")

    golden_rows = read_jsonl(GOLDEN_EVAL_PATH)
    child_chunks = read_jsonl(CHILD_CHUNKS_PATH)

    print("====== 加载数据 ======")
    print("golden_query_count:", len(golden_rows))
    print("child_chunk_count:", len(child_chunks))

    print("====== 构建 BM25 child index ======")
    bm25 = BM25Index(child_chunks)

    result_rows = []
    bad_cases = []

    metric_lists = {}
    for k in TOP_K_VALUES:
        metric_lists[f"hit@{k}"] = []
        metric_lists[f"recall@{k}"] = []
        metric_lists[f"mrr@{k}"] = []

    for row in golden_rows:
        query_id = row["query_id"]
        query = row["query"]
        gold_chunk_ids = row.get("gold_chunk_ids", [])

        retrieved = bm25.search(query, top_k=MAX_RETURN_K)
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
            "ground_truth": row.get("ground_truth"),
            "question_type": row.get("question_type"),
            "difficulty": row.get("difficulty"),
            "gold_parent_ids": row.get("gold_parent_ids", []),
            "gold_chunk_ids": gold_chunk_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_top_k": retrieved,
            "metrics": per_query_metrics,
        }

        result_rows.append(result_row)

        if per_query_metrics.get("hit@10", 0.0) == 0.0:
            bad_cases.append(result_row)

    aggregate_metrics = {
        "method": "bm25_child",
        "golden_query_count": len(golden_rows),
        "child_chunk_count": len(child_chunks),
        "top_k_values": TOP_K_VALUES,
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
    report_lines.append("# BM25 Child Retrieval Report")
    report_lines.append("")
    report_lines.append("## 1. Method")
    report_lines.append("")
    report_lines.append("- method: `BM25 over child_chunks`")
    report_lines.append(f"- bm25_k1: `{BM25_K1}`")
    report_lines.append(f"- bm25_b: `{BM25_B}`")
    report_lines.append("")
    report_lines.append("## 2. Data")
    report_lines.append("")
    report_lines.append(f"- golden_query_count: `{len(golden_rows)}`")
    report_lines.append(f"- child_chunk_count: `{len(child_chunks)}`")
    report_lines.append("")
    report_lines.append("## 3. Retrieval metrics")
    report_lines.append("")
    report_lines.append("| Metric | Value |")
    report_lines.append("|---|---:|")

    for k in TOP_K_VALUES:
        report_lines.append(f"| Hit@{k} | {aggregate_metrics[f'hit@{k}']:.4f} |")
        report_lines.append(f"| Recall@{k} | {aggregate_metrics[f'recall@{k}']:.4f} |")
        report_lines.append(f"| MRR@{k} | {aggregate_metrics[f'mrr@{k}']:.4f} |")

    report_lines.append("")
    report_lines.append("## 4. Bad cases")
    report_lines.append("")
    report_lines.append(f"- no_hit_at_10_count: `{len(bad_cases)}`")
    report_lines.append(f"- bad_cases_file: `{BAD_CASES_PATH.relative_to(PROJECT_ROOT)}`")
    report_lines.append("")
    report_lines.append("## 5. Interpretation")
    report_lines.append("")
    report_lines.append("- This is the first sparse retrieval baseline.")
    report_lines.append("- It evaluates whether BM25 can retrieve the gold child chunks from the full child chunk corpus.")
    report_lines.append("- Later parent-child expansion, dense retrieval, hybrid RRF, and reranking should be compared against this baseline.")
    report_lines.append("")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("====== BM25 child retrieval 完成 ======")
    print("results:", RESULTS_PATH)
    print("bad_cases:", BAD_CASES_PATH)
    print("metrics:", METRICS_PATH)
    print("report:", REPORT_PATH)

    print("====== 核心指标 ======")
    for k in TOP_K_VALUES:
        print(f"Hit@{k}: {aggregate_metrics[f'hit@{k}']:.4f}")
        print(f"Recall@{k}: {aggregate_metrics[f'recall@{k}']:.4f}")
        print(f"MRR@{k}: {aggregate_metrics[f'mrr@{k}']:.4f}")


if __name__ == "__main__":
    main()