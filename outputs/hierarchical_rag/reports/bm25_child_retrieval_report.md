# BM25 Child Retrieval Report

## 1. Method

- method: `BM25 over child_chunks`
- bm25_k1: `1.5`
- bm25_b: `0.75`

## 2. Data

- golden_query_count: `50`
- child_chunk_count: `944`

## 3. Retrieval metrics

| Metric | Value |
|---|---:|
| Hit@1 | 0.6800 |
| Recall@1 | 0.3100 |
| MRR@1 | 0.6800 |
| Hit@3 | 0.7800 |
| Recall@3 | 0.4700 |
| MRR@3 | 0.7267 |
| Hit@5 | 0.9200 |
| Recall@5 | 0.5700 |
| MRR@5 | 0.7587 |
| Hit@10 | 0.9600 |
| Recall@10 | 0.7700 |
| MRR@10 | 0.7642 |

## 4. Bad cases

- no_hit_at_10_count: `2`
- bad_cases_file: `outputs\hierarchical_rag\retrieval\bm25_child_bad_cases.jsonl`

## 5. Interpretation

- This is the first sparse retrieval baseline.
- It evaluates whether BM25 can retrieve the gold child chunks from the full child chunk corpus.
- Later parent-child expansion, dense retrieval, hybrid RRF, and reranking should be compared against this baseline.
