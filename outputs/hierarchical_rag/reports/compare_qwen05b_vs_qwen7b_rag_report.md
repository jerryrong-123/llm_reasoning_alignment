# Qwen2.5-0.5B vs Qwen2.5-7B RAG Generation Comparison

## 1. Purpose

This report compares the weak local generator Qwen2.5-0.5B-Instruct with the stronger server-side Qwen2.5-7B-Instruct on the same RAG context packs.

The goal is to determine whether low answer quality is mainly caused by retrieval/context issues or by generator capability.

## 2. Main Metrics

| Variant | 0.5B EM | 7B EM | Δ EM | 0.5B Contains | 7B Contains | Δ Contains | 0.5B Groundedness | 7B Groundedness | Δ Groundedness | 0.5B Soft Triad | 7B Soft Triad | Δ Soft Triad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top10_original_recheck | 0.1600 | 0.6400 | +0.4800 | 0.1800 | 0.7400 | +0.5600 | 0.5200 | 0.9600 | +0.4400 | 0.1800 | 0.7400 | +0.5600 |
| top10_soft_cap2_compressed | 0.1600 | 0.6400 | +0.4800 | 0.2000 | 0.7200 | +0.5200 | 0.4400 | 0.9400 | +0.5000 | 0.2000 | 0.7200 | +0.5200 |
| top7_soft_cap2_compressed | 0.1000 | 0.6200 | +0.5200 | 0.1200 | 0.7200 | +0.6000 | 0.3800 | 0.9400 | +0.5600 | 0.1200 | 0.7200 | +0.6000 |

## 3. Error Distribution

### top10_original_recheck

**Qwen2.5-0.5B errors:**

- ungrounded_generation: 23
- exact_correct: 8
- grounded_but_wrong: 16
- retrieval_context_missing_answer: 2
- partial_or_format_correct: 1

**Qwen2.5-7B errors:**

- exact_correct: 32
- partial_or_format_correct: 5
- grounded_but_wrong: 9
- ungrounded_generation: 2
- retrieval_context_missing_answer: 2

### top10_soft_cap2_compressed

**Qwen2.5-0.5B errors:**

- ungrounded_generation: 26
- exact_correct: 8
- grounded_but_wrong: 11
- retrieval_context_missing_answer: 3
- partial_or_format_correct: 2

**Qwen2.5-7B errors:**

- exact_correct: 32
- partial_or_format_correct: 4
- grounded_but_wrong: 9
- ungrounded_generation: 2
- retrieval_context_missing_answer: 3

### top7_soft_cap2_compressed

**Qwen2.5-0.5B errors:**

- ungrounded_generation: 30
- grounded_but_wrong: 11
- exact_correct: 5
- retrieval_context_missing_answer: 3
- partial_or_format_correct: 1

**Qwen2.5-7B errors:**

- exact_correct: 31
- partial_or_format_correct: 5
- grounded_but_wrong: 9
- ungrounded_generation: 2
- retrieval_context_missing_answer: 3

## 4. Conclusion

Qwen2.5-7B substantially improves answer generation over Qwen2.5-0.5B under the same retrieved contexts.

The strongest 7B result reaches EM=0.6400, Contains=0.7400, Groundedness=0.9600, and Soft Triad Pass=0.7400 on the Top10 original context pack.

This shows that the previous low answer quality was mainly caused by the weak 0.5B generator rather than by retrieval failure, because context recall and answerability were already high.

The next step should use Qwen2.5-7B as the no-finetuning baseline before running LoRA SFT, so that any fine-tuning gain can be measured against a strong open-source generator.
