# RAG Answer Generation v4 Balanced Prompt Report

## 1. Purpose

This experiment tests whether a softer balanced prompt fixes the yes/no and not-supported collapse observed in v3 generation.

## 2. Metrics

| Variant | EM | Contains | Context Recall | Context Precision | Groundedness | Answerability | Strict Triad | Soft Triad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| top10_original_recheck | 0.1600 | 0.1800 | 0.9467 | 0.2100 | 0.5200 | 0.9400 | 0.1600 | 0.1800 |
| top10_soft_cap2_compressed | 0.1600 | 0.2000 | 0.9467 | 0.2171 | 0.4400 | 0.9200 | 0.1600 | 0.2000 |
| top7_soft_cap2_compressed | 0.1000 | 0.1200 | 0.9233 | 0.2914 | 0.3800 | 0.9200 | 0.1000 | 0.1200 |

## 3. Interpretation

- If v4 improves over v3 but remains below the old baseline, prompt collapse was one issue but Qwen2.5-0.5B remains the main bottleneck.
- If v4 recovers the old baseline, Context Pack v3 can be combined with a balanced prompt.
- If v4 still performs poorly, the next step should move to a stronger generator on the rented server.
