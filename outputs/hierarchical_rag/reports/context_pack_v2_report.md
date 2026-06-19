# Context Pack v2 Report

## 1. Purpose

This step optimizes the final RAG context pack by applying parent-level deduplication, TopK selection, and sentence-level compression.

It does not use gold answers for compression, so it avoids answer leakage.

## 2. Metrics

| Variant | Hit | Recall | Precision | MRR | Relevance Mean | Avg Context Count | Avg Total Chars |
|---|---:|---:|---:|---:|---:|---:|---:|
| top5_parent_dedup | 1.0000 | 0.8433 | 0.3650 | 0.9640 | 0.4132 | 4.98 | 1721.26 |
| top7_parent_dedup | 1.0000 | 0.8700 | 0.2764 | 0.9640 | 0.3992 | 6.82 | 2382.18 |
| top7_compressed | 1.0000 | 0.8700 | 0.2764 | 0.9640 | 0.3910 | 6.82 | 1652.16 |
| top10_compressed | 1.0000 | 0.8800 | 0.2492 | 0.9640 | 0.3877 | 7.78 | 1896.96 |

## 3. Interpretation

- Parent deduplication reduces repeated chunks from the same source document.
- Top7 is introduced as a balance between Top5 recall loss and Top10 noise.
- Sentence-level compression reduces context length and should help weaker generators avoid distractors.
- The next step should run answer generation on these v2 context packs and compare EM, Contains Match, Groundedness, and Triad Pass.
