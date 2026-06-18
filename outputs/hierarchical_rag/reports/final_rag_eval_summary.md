# Final Hierarchical RAG Evaluation Summary

## 1. Retrieval Summary

| Stage | Hit@10 | Recall@10 | MRR@10 | Note |
|---|---:|---:|---:|---|
| BM25 child | 0.9600 | 0.7700 | 0.7642 | Sparse lexical retrieval baseline. |
| Parent BM25 + child expansion | 1.0000 | 0.9000 | 0.8057 | Hierarchical parent retrieval followed by child expansion. |
| BGE embedding child | 1.0000 | 0.9300 | 0.9640 | Dense semantic retrieval. |
| Default Hybrid RRF | 1.0000 | 0.8667 | 0.8822 | Initial weighted RRF fusion. |
| Best Hybrid RRF | 1.0000 | 0.9467 | 0.9123 | Grid-searched RRF weights. |
| Best Hybrid RRF + BGE rerank | 1.0000 | 0.9467 | 0.9640 | Final retrieval pipeline. |

## 2. Final RAG Context Pack

| Setting | Hit | Recall | Precision | MRR |
|---|---:|---:|---:|---:|
| Top5 | 1.0000 | 0.8600 | 0.3800 | 0.9640 |
| Top10 | 1.0000 | 0.9467 | 0.2100 | 0.9640 |

## 3. Answer Generation Summary

| Setting | Exact Match | Contains Match | Avg Context Recall |
|---|---:|---:|---:|
| Top5 | 0.1200 | 0.5000 | 0.8600 |
| Top10 | 0.1400 | 0.4600 | 0.9467 |

## 4. RAG Triad Proxy Summary

| Setting | Groundedness Proxy | Answerability Proxy | Soft Triad Pass |
|---|---:|---:|---:|
| Top5 | 0.6400 | 0.8600 | 0.3400 |
| Top10 | 0.7800 | 0.9400 | 0.3600 |

### Top10 Error Categories

- exact_correct: `7`
- grounded_but_wrong: `19`
- partial_or_format_correct: `16`
- retrieval_context_missing_answer: `2`
- ungrounded_generation: `6`

## 5. Local LLM-as-a-Judge Summary

| Setting | LLM Correctness | LLM Groundedness | Note |
|---|---:|---:|---|
| Top5 | 0.1000 | 0.1000 | 0.5B local judge, approximate only |
| Top10 | 0.1200 | 0.1200 | 0.5B local judge, approximate only |

## 6. Final Conclusions

- Retrieval optimization is effective: BM25 child Recall@10 improves to the final Best Hybrid RRF + BGE rerank Recall@10 of 0.9467.
- Reranking improves evidence ordering: the final retrieval pipeline reaches MRR@10 of 0.9640.
- Final context answerability is high, but answer generation with Qwen2.5-0.5B remains weak, indicating the bottleneck shifts from retrieval to generation.
- Top10 has higher context recall than Top5, but also lower precision and more noise.
- Local LLM-as-a-Judge is useful as a low-cost evaluation framework demo, but the 0.5B judge is not reliable enough for final authority-level evaluation.
