# V3 Generation Failure Diagnostics

## 1. Purpose

This report diagnoses why Context Pack v3 plus extractive short-answer prompting failed to improve Qwen2.5-0.5B RAG generation.

## 2. Summary

| Variant | EM | Contains | Groundedness | Truncated Rate | Visible Answerability | Original Answerability | Avg Prompt Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| top10_original_recheck | 0.0600 | 0.0800 | 0.1600 | 0.0000 | 0.9600 | 0.9400 | 1152.18 |
| top10_soft_cap2_compressed | 0.0800 | 0.0800 | 0.1600 | 0.0000 | 0.9400 | 0.9200 | 980.06 |
| top7_soft_cap2_compressed | 0.0600 | 0.0600 | 0.0800 | 0.0000 | 0.9400 | 0.9200 | 755.00 |

## 3. Prediction Type Counts

### top10_original_recheck

- short_phrase: 4
- single_token_or_entity: 2
- yes_no_yes: 27
- not_supported_like: 13
- yes_no_no: 4

### top10_soft_cap2_compressed

- short_phrase: 4
- single_token_or_entity: 1
- yes_no_yes: 27
- not_supported_like: 14
- yes_no_no: 4

### top7_soft_cap2_compressed

- short_phrase: 2
- single_token_or_entity: 4
- yes_no_yes: 32
- not_supported_like: 11
- yes_no_no: 1

## 4. Error Category Counts

### top10_original_recheck

- ungrounded_generation: 38
- exact_correct: 3
- grounded_but_wrong: 5
- partial_or_format_correct: 1
- retrieval_context_missing_answer: 3

### top10_soft_cap2_compressed

- ungrounded_generation: 38
- exact_correct: 4
- grounded_but_wrong: 5
- retrieval_context_missing_answer: 3

### top7_soft_cap2_compressed

- ungrounded_generation: 42
- exact_correct: 3
- grounded_but_wrong: 2
- retrieval_context_missing_answer: 3

## 5. Interpretation Guide

- If truncated rate is high and visible answerability is much lower than original answerability, the generation failure is partly caused by input truncation.
- If visible answerability is still high but EM and Contains are low, the main bottleneck is Qwen2.5-0.5B generation ability or prompt following.
- If predictions collapse into yes/no/not supported, the prompt is too restrictive or the model is too weak to follow extractive QA instructions.
- If groundedness is low mainly for yes/no answers, the proxy evaluator needs a yes/no-aware groundedness rule.
