# Final answer reward test

## Purpose

This report tests the reward function before using it in GRPO/RLVR.

The reward design follows:

```text
answer correctness reward > final-answer format reward
```

## Reward definition

```text
correctness_reward:
  +1.0 if extracted numeric answer equals gold answer

format_reward:
  +0.1 if response contains a final-answer format

extractability_reward:
  +0.1 if a numeric prediction can be extracted

total_reward = correctness_reward + format_reward + extractability_reward
```

## Summary

| metric | value |
|---|---:|
| total | 20 |
| correctness_count | 11 |
| correctness_rate | 0.5500 |
| format_hit_count | 19 |
| format_hit_rate | 0.9500 |
| extractable_count | 20 |
| extractable_rate | 1.0000 |
| avg_total_reward | 0.7450 |

## Case table

| doc_id | gold | pred | correct | format_hit | extractable | total_reward |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 18 | -10 | 0 | 1 | 1 | 0.2 |
| 1 | 3 | 3 | 1 | 1 | 1 | 1.2 |
| 2 | 70000 | 70,000 | 1 | 1 | 1 | 1.2 |
| 3 | 540 | 540 | 1 | 1 | 1 | 1.2 |
| 4 | 20 | 20 | 1 | 1 | 1 | 1.2 |
| 5 | 64 | 44 | 0 | 1 | 1 | 0.2 |
| 6 | 260 | 260 | 1 | 1 | 1 | 1.2 |
| 7 | 160 | 120 | 0 | 1 | 1 | 0.2 |
| 8 | 45 | -160 | 0 | 1 | 1 | 0.2 |
| 9 | 460 | 460 | 1 | 1 | 1 | 1.2 |
| 10 | 366 | 294 | 0 | 1 | 1 | 0.2 |
| 11 | 694 | 2233 | 0 | 1 | 1 | 0.2 |
| 12 | 13 | 10 | 0 | 1 | 1 | 0.2 |
| 13 | 18 | 10 | 0 | 1 | 1 | 0.2 |
| 14 | 60 | 60 | 1 | 1 | 1 | 1.2 |
| 15 | 125 | 125 | 1 | 1 | 1 | 1.2 |
| 16 | 230 | 230 | 1 | 1 | 1 | 1.2 |
| 17 | 57500 | 57500 | 1 | 1 | 1 | 1.2 |
| 18 | 7 | 7 | 1 | 1 | 1 | 1.2 |
| 19 | 6 | 12 | 0 | 0 | 1 | 0.1 |
