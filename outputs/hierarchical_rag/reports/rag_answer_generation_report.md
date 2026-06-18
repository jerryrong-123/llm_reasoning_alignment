# RAG Answer Generation Report

## 1. Method

- model: `Qwen2.5-0.5B-Instruct-local`
- retriever: `best_hybrid_rrf_plus_bge_rerank`
- generation mode: `greedy decoding`

## 2. Metrics

| Setting | Exact Match | Contains Match | Avg Context Recall | Avg Context MRR |
|---|---:|---:|---:|---:|
| Top5 | 0.1200 | 0.5000 | 0.8600 | 0.9640 |
| Top10 | 0.1400 | 0.4600 | 0.9467 | 0.9640 |

## 3. Interpretation

- Top5 has less noise but lower evidence recall.
- Top10 has higher evidence recall but may include more distractor contexts.
- Exact Match is strict; Contains Match is a softer signal for short-answer QA.
