# Hybrid RRF Grid Search Report

## 1. Purpose

This experiment searches RRF weights because the default Hybrid RRF configuration underperformed the embedding-only retriever.

## 2. Search space

- RRF_K_VALUES: `[10, 30, 60, 100]`
- BM25_WEIGHTS: `[0.0, 0.25, 0.5, 1.0]`
- EMBEDDING_WEIGHTS: `[1.0, 2.0, 3.0, 5.0]`
- PARENT_WEIGHTS: `[0.0, 0.25, 0.5, 0.8, 1.0]`
- searched_config_count: `320`

## 3. Best config

- rrf_k: `30`
- bm25_weight: `0.0`
- embedding_weight: `1.0`
- parent_weight: `1.0`

## 4. Best metrics

| Metric | Value |
|---|---:|
| Hit@1 | 0.8600 |
| Recall@1 | 0.3967 |
| MRR@1 | 0.8600 |
| Hit@3 | 0.9600 |
| Recall@3 | 0.6733 |
| MRR@3 | 0.9033 |
| Hit@5 | 1.0000 |
| Recall@5 | 0.8000 |
| MRR@5 | 0.9123 |
| Hit@10 | 1.0000 |
| Recall@10 | 0.9467 |
| MRR@10 | 0.9123 |

## 5. Top 10 configs

| Rank | rrf_k | bm25 | embedding | parent | Recall@10 | Hit@10 | MRR@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30 | 0.0 | 1.0 | 1.0 | 0.9467 | 1.0000 | 0.9123 |
| 2 | 60 | 0.0 | 1.0 | 1.0 | 0.9467 | 1.0000 | 0.9123 |
| 3 | 100 | 0.0 | 1.0 | 1.0 | 0.9467 | 1.0000 | 0.9123 |
| 4 | 10 | 0.0 | 1.0 | 1.0 | 0.9467 | 1.0000 | 0.9113 |
| 5 | 10 | 0.25 | 1.0 | 0.8 | 0.9467 | 1.0000 | 0.9013 |
| 6 | 10 | 1.0 | 2.0 | 0.8 | 0.9400 | 1.0000 | 0.9170 |
| 7 | 10 | 0.5 | 1.0 | 0.5 | 0.9400 | 1.0000 | 0.9170 |
| 8 | 10 | 1.0 | 2.0 | 1.0 | 0.9400 | 1.0000 | 0.9170 |
| 9 | 30 | 1.0 | 2.0 | 0.8 | 0.9400 | 1.0000 | 0.9170 |
| 10 | 30 | 0.5 | 1.0 | 0.5 | 0.9400 | 1.0000 | 0.9163 |

## 6. Interpretation

- If the best config is embedding-heavy, it means dense retrieval is the strongest signal on this benchmark.
- If BM25 or parent weights hurt Recall@10, they should be treated as ablation components rather than the final retriever.
- This gives a defensible experiment instead of blindly claiming hybrid retrieval is always better.
