# BGE Rerank over Best Hybrid RRF Candidates

## 1. Method

- base retriever: `best Hybrid RRF`
- reranker: `local BGE bi-encoder similarity`
- candidate set: `Hybrid RRF Top10`

## 2. Metrics

| Metric | Base Hybrid | BGE Rerank |
|---|---:|---:|
| Hit@1 | 0.8600 | 0.9400 |
| Recall@1 | 0.3967 | 0.4333 |
| MRR@1 | 0.8600 | 0.9400 |
| Hit@3 | 0.9600 | 0.9800 |
| Recall@3 | 0.6733 | 0.7833 |
| MRR@3 | 0.9033 | 0.9600 |
| Hit@5 | 1.0000 | 1.0000 |
| Recall@5 | 0.8000 | 0.8600 |
| MRR@5 | 0.9123 | 0.9640 |
| Hit@10 | 1.0000 | 1.0000 |
| Recall@10 | 0.9467 | 0.9467 |
| MRR@10 | 0.9123 | 0.9640 |

## 3. Interpretation

- Reranking does not change the candidate pool, so Recall@10 usually stays close to the base Hybrid RRF.
- The main goal is to improve top-rank quality such as Hit@1 and MRR.
- If reranking hurts MRR, the final system should keep Best Hybrid RRF as retriever and use reranking only as an ablation.
