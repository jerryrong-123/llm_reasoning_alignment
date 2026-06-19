# Context Pack v3 cap3 Report

## 1. Purpose

Context Pack v2 showed that hard parent deduplication reduced evidence recall. This v3 experiment uses soft parent cap=3 and compression-only variants to balance recall, precision, answerability, and context length.

Compression uses only query/title information, not ground truth answers, so it avoids answer leakage.

## 2. Metrics

| Variant | Hit | Recall | Precision | MRR | Relevance Mean | Answerability | Avg Count | Avg Total Chars |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| top10_original_recheck | 1.0000 | 0.9467 | 0.2100 | 0.9640 | 0.3833 | 0.9400 | 10.00 | 3473.70 |
| top10_compressed_only | 1.0000 | 0.9467 | 0.2100 | 0.9640 | 0.3827 | 0.9200 | 10.00 | 2913.12 |
| top10_soft_cap3 | 1.0000 | 0.9467 | 0.2107 | 0.9640 | 0.3836 | 0.9400 | 9.98 | 3467.90 |
| top10_soft_cap3_compressed | 1.0000 | 0.9467 | 0.2107 | 0.9640 | 0.3830 | 0.9200 | 9.98 | 2909.36 |
| top7_soft_cap3_compressed | 1.0000 | 0.9233 | 0.2914 | 0.9640 | 0.3938 | 0.9200 | 7.00 | 2008.98 |

## 3. Interpretation

- `top10_original_recheck` is the baseline recheck for comparison.
- `top10_compressed_only` keeps the same chunk set but compresses text, so chunk-level recall should stay high while context length drops.
- `top10_soft_cap3` reduces repeated chunks from the same parent while allowing up to two chunks per parent.
- `top10_soft_cap3_compressed` combines soft parent cap and compression.
- `top7_soft_cap3_compressed` tests a shorter context budget for weaker generators.

A good v3 pack should preserve high recall and answerability while reducing context length and noise.
