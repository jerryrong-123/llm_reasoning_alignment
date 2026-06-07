# Prompt format v2 reward diagnosis

## Purpose

This report diagnoses the prompt-format-v2 evaluation results before moving to reward-based format optimization.

The goal is to decide how to combine answer correctness reward and final-answer format reward.

## Category summary

| category | count | ratio |
|---|---:|---:|
| answer_correct_and_format_correct | 8 | 0.4000 |
| answer_correct_but_format_missing | 3 | 0.1500 |
| answer_wrong_and_format_missing | 3 | 0.1500 |
| answer_wrong_but_format_hit | 6 | 0.3000 |

## Key conclusion

The reward design should follow this priority:

```text
answer correctness reward > final-answer format reward
```

Reason:

1. Some samples have correct answers but imperfect final-answer formatting.
2. Some samples may satisfy the final-answer format while still producing wrong answers.
3. Therefore, a format-only reward is unsafe.
4. Format reward should be a small auxiliary reward, not the main optimization target.

## Suggested reward structure

```text
total_reward = correctness_reward + small_format_reward + small_extractability_reward
```

Suggested weights for the next debug experiment:

```text
correctness_reward:
  +1.0 if final numeric answer is correct
  0.0 otherwise

format_reward:
  +0.1 if the response contains a final answer line
  0.0 otherwise

extractability_reward:
  +0.1 if a numeric answer can be extracted
  0.0 otherwise
```

This keeps the maximum auxiliary format reward at 0.2, much smaller than the correctness reward.

## Case table

| doc_id | category | gold | flexible_pred | final_answer_pred |
|---:|---|---:|---:|---:|
| 0 | answer_wrong_and_format_missing | 18 | -10 | None |
| 1 | answer_correct_and_format_correct | 3 | 3 | 3 |
| 2 | answer_correct_but_format_missing | 70000 | 70,000 | None |
| 3 | answer_correct_and_format_correct | 540 | 540 | 540 |
| 4 | answer_correct_and_format_correct | 20 | 20 | 20 |
| 5 | answer_wrong_but_format_hit | 64 | 44 | 44 |
| 6 | answer_correct_and_format_correct | 260 | 260 | 260 |
| 7 | answer_wrong_but_format_hit | 160 | 120 | 120 |
| 8 | answer_wrong_but_format_hit | 45 | -160 | -160 |
| 9 | answer_correct_but_format_missing | 460 | 460 | None |
| 10 | answer_wrong_but_format_hit | 366 | 294 | 294 |
| 11 | answer_wrong_but_format_hit | 694 | 2233 | 2233 |
| 12 | answer_wrong_but_format_hit | 13 | 10 | 10 |
| 13 | answer_wrong_and_format_missing | 18 | 10 | None |
| 14 | answer_correct_and_format_correct | 60 | 60 | 60 |
| 15 | answer_correct_but_format_missing | 125 | 125 | None |
| 16 | answer_correct_and_format_correct | 230 | 230 | 230 |
| 17 | answer_correct_and_format_correct | 57500 | 57500 | 57500 |
| 18 | answer_correct_and_format_correct | 7 | 7 | 7 |
| 19 | answer_wrong_and_format_missing | 6 | 12 | None |
