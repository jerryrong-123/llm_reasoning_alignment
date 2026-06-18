# Hybrid RRF Retrieval Report

## 1. Method

- method: `Reciprocal Rank Fusion`
- rrf_k: `60`
- source_weights: `{'bm25_child': 1.0, 'embedding_child': 1.0, 'parent_expansion': 0.8}`

## 2. Sources

- BM25 child retrieval
- Embedding child retrieval
- Parent BM25 + child expansion

## 3. Current Hybrid RRF metrics

| Metric | Value |
|---|---:|
| Hit@1 | 0.8200 |
| Recall@1 | 0.3767 |
| MRR@1 | 0.8200 |
| Hit@3 | 0.9200 |
| Recall@3 | 0.6000 |
| MRR@3 | 0.8667 |
| Hit@5 | 0.9600 |
| Recall@5 | 0.7000 |
| MRR@5 | 0.8767 |
| Hit@10 | 1.0000 |
| Recall@10 | 0.8667 |
| MRR@10 | 0.8822 |

## 4. Previous baseline comparison

| Method | Recall@10 |
|---|---:|
| BM25 child | 0.7700 |
| Embedding child | 0.9300 |
| Parent expansion | 0.9000 |
| Hybrid RRF | 0.8667 |

## 5. Interpretation

- Hybrid RRF combines sparse lexical matching, dense semantic matching, and parent-level expansion.
- If Hybrid RRF improves Recall@10, it becomes the best retriever candidate before reranking.
- If it does not improve over embedding alone, the project can still report that BGE embedding is already strong and RRF adds limited value on this 50-query benchmark.
