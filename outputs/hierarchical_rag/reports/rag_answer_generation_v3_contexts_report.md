# RAG Answer Generation with Context Pack v3 Report

## 1. Purpose

This experiment evaluates whether Context Pack v3 and extractive short-answer prompting improve RAG generation quality.

## 2. Metrics

| Variant | EM | Contains | Context Recall | Context Precision | Groundedness | Answerability | Strict Triad | Soft Triad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| top10_original_recheck | 0.0600 | 0.0800 | 0.9467 | 0.2100 | 0.1600 | 0.9400 | 0.0600 | 0.0600 |
| top10_soft_cap2_compressed | 0.0800 | 0.0800 | 0.9467 | 0.2171 | 0.1600 | 0.9200 | 0.0600 | 0.0600 |
| top7_soft_cap2_compressed | 0.0600 | 0.0600 | 0.9233 | 0.2914 | 0.0800 | 0.9200 | 0.0400 | 0.0400 |

## 3. Interpretation

- `top10_original_recheck` tests prompt improvement on the original final context pack.
- `top10_soft_cap2_compressed` keeps high recall while reducing repeated parent evidence and context length.
- `top7_soft_cap2_compressed` tests whether weaker generators benefit from shorter, cleaner contexts.
- If EM and Contains do not improve, the main bottleneck is likely the 0.5B generator rather than context packing.
