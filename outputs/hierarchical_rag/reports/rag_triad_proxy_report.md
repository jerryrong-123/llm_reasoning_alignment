# RAG Triad Proxy Evaluation Report

## 1. Purpose

This step provides a low-cost diagnostic approximation of the RAG Triad:

- Context relevance
- Groundedness
- Answer correctness

It does not call an LLM judge. It is used for quick local diagnosis before LLM-as-a-Judge or RAGAS.

## 2. Metrics

| Setting | EM | Contains | Context Recall | Context MRR | Groundedness Proxy | Answerability Proxy | Strict Triad Pass | Soft Triad Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| top5 | 0.1200 | 0.5000 | 0.8600 | 0.9640 | 0.6400 | 0.8600 | 0.1000 | 0.3400 |
| top10 | 0.1400 | 0.4600 | 0.9467 | 0.9640 | 0.7800 | 0.9400 | 0.1400 | 0.3600 |

## 3. Error categories

### Top5

- exact_correct: `6`
- grounded_but_wrong: `11`
- partial_or_format_correct: `19`
- retrieval_context_missing_answer: `5`
- ungrounded_generation: `9`

### Top10

- exact_correct: `7`
- grounded_but_wrong: `19`
- partial_or_format_correct: `16`
- retrieval_context_missing_answer: `2`
- ungrounded_generation: `6`

## 4. Interpretation

- If context recall is high but answer correctness is low, the bottleneck is generation rather than retrieval.
- If answerability proxy is low, the retrieved contexts may not directly contain the final answer string.
- If groundedness proxy is low, the model may be generating unsupported answers.
- Exact Match is strict; Contains Match and Soft Triad Pass are useful secondary signals.
