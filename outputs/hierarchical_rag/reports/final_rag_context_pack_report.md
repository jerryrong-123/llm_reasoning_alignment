# Final RAG Context Pack Report

## 1. Purpose

This step converts the final reranked retrieval results into RAG-ready input files.

## 2. Input

- golden_eval: `data\processed\hierarchical_rag\golden_eval_50.jsonl`
- rerank_results: `outputs\hierarchical_rag\retrieval\bge_rerank_hybrid_results.jsonl`

## 3. Output

- rag_inputs_top5: `data\processed\hierarchical_rag\rag_inputs_top5.jsonl`
- rag_inputs_top10: `data\processed\hierarchical_rag\rag_inputs_top10.jsonl`
- context_metrics: `outputs\hierarchical_rag\eval\final_rag_context_metrics.json`

## 4. Context metrics

| Metric | Value |
|---|---:|
| Context Hit@5 | 1.0000 |
| Context Recall@5 | 0.8600 |
| Context Precision@5 | 0.3800 |
| Context MRR@5 | 0.9640 |
| Avg Context Count@5 | 5.00 |
| Context Hit@10 | 1.0000 |
| Context Recall@10 | 0.9467 |
| Context Precision@10 | 0.2100 |
| Context MRR@10 | 0.9640 |
| Avg Context Count@10 | 10.00 |

## 5. Interpretation

- Top5 is more concise and usually better for answer generation when context noise matters.
- Top10 has higher evidence recall and is safer for multi-hop questions.
- The next step will generate answers using these packed contexts.
