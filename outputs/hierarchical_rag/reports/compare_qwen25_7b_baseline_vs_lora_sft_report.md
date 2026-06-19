# Qwen2.5-7B Baseline vs Qwen2.5-7B LoRA SFT RAG Comparison

## 1. Purpose

This report compares the no-finetuning Qwen2.5-7B RAG baseline with the Qwen2.5-7B LoRA SFT model on the same 50-example RAG evaluation set.

The goal is to determine whether RAG-SFT improves answer quality beyond the strong 7B baseline.

## 2. Main Metrics

| Variant | Base EM | SFT EM | Δ EM | Base Contains | SFT Contains | Δ Contains | Base Grounded | SFT Grounded | Δ Grounded | Base SoftTriad | SFT SoftTriad | Δ SoftTriad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top10_original_recheck | 0.6400 | 0.6400 | +0.0000 | 0.7400 | 0.8000 | +0.0600 | 0.9600 | 0.9800 | +0.0200 | 0.7400 | 0.8000 | +0.0600 |
| top10_soft_cap2_compressed | 0.6400 | 0.6400 | +0.0000 | 0.7200 | 0.8000 | +0.0800 | 0.9400 | 0.9600 | +0.0200 | 0.7200 | 0.7800 | +0.0600 |
| top7_soft_cap2_compressed | 0.6200 | 0.6600 | +0.0400 | 0.7200 | 0.8200 | +0.1000 | 0.9400 | 0.9600 | +0.0200 | 0.7200 | 0.8000 | +0.0800 |

## 3. Error Distribution

### top10_original_recheck

**Baseline errors:**
- exact_correct: 32
- partial_or_format_correct: 5
- grounded_but_wrong: 9
- ungrounded_generation: 2
- retrieval_context_missing_answer: 2

**LoRA SFT errors:**
- exact_correct: 32
- partial_or_format_correct: 8
- grounded_but_wrong: 7
- retrieval_context_missing_answer: 3

### top10_soft_cap2_compressed

**Baseline errors:**
- exact_correct: 32
- partial_or_format_correct: 4
- grounded_but_wrong: 9
- ungrounded_generation: 2
- retrieval_context_missing_answer: 3

**LoRA SFT errors:**
- exact_correct: 32
- partial_or_format_correct: 8
- grounded_but_wrong: 7
- retrieval_context_missing_answer: 3

### top7_soft_cap2_compressed

**Baseline errors:**
- exact_correct: 31
- partial_or_format_correct: 5
- grounded_but_wrong: 9
- ungrounded_generation: 2
- retrieval_context_missing_answer: 3

**LoRA SFT errors:**
- exact_correct: 33
- partial_or_format_correct: 8
- grounded_but_wrong: 6
- retrieval_context_missing_answer: 3

## 4. Conclusion

LoRA SFT does not substantially improve exact match on the Top10 variants, but it improves contains match, groundedness, and soft triad pass.

The best post-SFT result is obtained by `top7_soft_cap2_compressed`, reaching EM=0.6600, Contains=0.8200, Groundedness=0.9600, and SoftTriad=0.8000.

This suggests that RAG-SFT mainly improves answer style and context-grounded answer extraction rather than dramatically changing exact-match correctness.
