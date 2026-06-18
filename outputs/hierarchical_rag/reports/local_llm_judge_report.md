# Local LLM-as-a-Judge Report

## 1. Purpose

This step uses a local Qwen2.5-0.5B-Instruct model as a lightweight judge.
It is intended for low-cost CI-style evaluation, not as a high-accuracy release judge.

## 2. Metrics

| Setting | LLM Correctness | LLM Groundedness | LLM Context Relevance | LLM Answer Quality | EM | Contains | Context Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| top5 | 0.1000 | 0.1000 | 5.0000 | 5.0000 | 0.1200 | 0.5000 | 0.8600 |
| top10 | 0.1200 | 0.1200 | 5.0000 | 5.0000 | 0.1400 | 0.4600 | 0.9467 |

## 3. Error type counts

### Top5

- unknown: `5`
- wrong_reasoning: `45`

### Top10

- unknown: `6`
- wrong_reasoning: `44`

## 4. Interpretation

- This local judge provides a richer signal than exact match.
- Because the judge is a small local model, results should be treated as approximate.
- A stronger release judge can be added later using a larger LLM or RAGAS.
