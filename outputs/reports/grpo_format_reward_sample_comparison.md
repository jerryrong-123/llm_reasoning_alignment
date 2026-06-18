# GRPO format reward sample comparison

## Purpose

This report compares exact lm-eval sample files for:

```text
sft_lora_small_v2
grpo_lora_small
grpo_lora_small_format_reward
```

This script uses exact eval output directories and exact checkpoint sample directories.
It intentionally rejects format/prompt variants unless they are explicitly requested.

## Exact sample file check

| model | eval_output_dir | checkpoint_dir | sample_file |
|---|---|---|---|
| sft_lora_small_v2 | sft_lora_small_v2_qwen25_15b_gsm8k_cot_limit20 | outputs__checkpoints__sft_lora_small_v2 | outputs/eval/sft_lora_small_v2_qwen25_15b_gsm8k_cot_limit20/outputs__checkpoints__sft_lora_small_v2/samples_gsm8k_cot_2026-06-07T19-15-00.209160.jsonl |
| grpo_lora_small | grpo_lora_small_qwen25_15b_gsm8k_cot_limit20 | outputs__checkpoints__grpo_lora_small | outputs/eval/grpo_lora_small_qwen25_15b_gsm8k_cot_limit20/outputs__checkpoints__grpo_lora_small/samples_gsm8k_cot_2026-06-07T01-56-24.175100.jsonl |
| grpo_lora_small_format_reward | grpo_lora_small_format_reward_qwen25_15b_gsm8k_cot_limit20 | outputs__checkpoints__grpo_lora_small_format_reward | outputs/eval/grpo_lora_small_format_reward_qwen25_15b_gsm8k_cot_limit20/outputs__checkpoints__grpo_lora_small_format_reward/samples_gsm8k_cot_2026-06-08T18-25-49.289819.jsonl |

## Accuracy summary from parsed samples

| model | correct | total | parsed_acc |
|---|---:|---:|---:|
| sft_lora_small_v2 | 12 | 20 | 0.6000 |
| grpo_lora_small | 9 | 20 | 0.4500 |
| grpo_lora_small_format_reward | 9 | 20 | 0.4500 |

## Category summary

| category | count |
|---|---:|
| format_reward_correct | 9 |
| regressed_from_sft_v2 | 3 |
| still_wrong | 8 |

## Interpretation

- The exact file check confirms that `sft_lora_small_v2` is not the `sft_lora_small_v2_format` variant.
- The format-reward GRPO checkpoint matches old `grpo_lora_small` on parsed sample accuracy.
- The format-reward GRPO checkpoint still underperforms `sft_lora_small_v2` on these parsed samples.
- This is sample-level error analysis, not a replacement for official lm-eval metrics.

## Case table

| doc | gold | sft_pred | sft_ok | old_pred | old_ok | fmt_pred | fmt_ok | category |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 18 | 64 | 0 | 96 | 0 | 88 | 0 | still_wrong |
| 1 | 3 | 3 | 1 | 3 | 1 | 3 | 1 | format_reward_correct |
| 2 | 70000 | 70000 | 1 | 110000 | 0 | 110000 | 0 | regressed_from_sft_v2 |
| 3 | 540 | 540 | 1 | 540 | 1 | 540 | 1 | format_reward_correct |
| 4 | 20 | 20 | 1 | 20 | 1 | 20 | 1 | format_reward_correct |
| 5 | 64 | 29 | 0 | 50 | 0 | 50 | 0 | still_wrong |
| 6 | 260 | 260 | 1 | 260 | 1 | 260 | 1 | format_reward_correct |
| 7 | 160 | 296 | 0 | 120 | 0 | 90 | 0 | still_wrong |
| 8 | 45 | 2 | 0 | 400 | 0 | 400 | 0 | still_wrong |
| 9 | 460 | 460 | 1 | 460 | 1 | 460 | 1 | format_reward_correct |
| 10 | 366 | 366 | 1 | 366 | 1 | 366 | 1 | format_reward_correct |
| 11 | 694 | 1588 | 0 | 8400 | 0 | 8400 | 0 | still_wrong |
| 12 | 13 | 63 | 0 | 9 | 0 | 9 | 0 | still_wrong |
| 13 | 18 | 36 | 0 | 17 | 0 | 24 | 0 | still_wrong |
| 14 | 60 | 60 | 1 | 60 | 1 | 60 | 1 | format_reward_correct |
| 15 | 125 | 2971 | 0 | 2971 | 0 | 2971 | 0 | still_wrong |
| 16 | 230 | 230 | 1 | 160 | 0 | 160 | 0 | regressed_from_sft_v2 |
| 17 | 57500 | 57500 | 1 | 57500 | 1 | 57500 | 1 | format_reward_correct |
| 18 | 7 | 7 | 1 | 7 | 1 | 7 | 1 | format_reward_correct |
| 19 | 6 | 6 | 1 | 2 | 0 | 2 | 0 | regressed_from_sft_v2 |
