# Embedding Child Retrieval Report

## 1. Method

- method: `embedding retrieval over child_chunks`
- loader: `transformers AutoTokenizer + AutoModel`
- embedding_model: `models\bge-small-en-v1.5`
- pooling: `CLS pooling`
- similarity: `cosine similarity via normalized dot product`

## 2. Data

- golden_query_count: `50`
- child_chunk_count: `944`

## 3. Retrieval metrics

| Metric | Value |
|---|---:|
| Hit@1 | 0.9400 |
| Recall@1 | 0.4333 |
| MRR@1 | 0.9400 |
| Hit@3 | 0.9800 |
| Recall@3 | 0.7833 |
| MRR@3 | 0.9600 |
| Hit@5 | 1.0000 |
| Recall@5 | 0.8600 |
| MRR@5 | 0.9640 |
| Hit@10 | 1.0000 |
| Recall@10 | 0.9300 |
| MRR@10 | 0.9640 |

## 4. Bad cases

- no_hit_at_10_count: `0`
- bad_cases_file: `outputs\hierarchical_rag\retrieval\embedding_child_bad_cases.jsonl`

## 5. Interpretation

- This dense retrieval baseline avoids sentence_transformers because the local Windows Python 3.12 environment crashes while importing pyarrow through sentence_transformers dependencies.
- This script directly uses transformers to load the local BGE model.
- Later Hybrid RRF will combine this dense retriever with BM25.
