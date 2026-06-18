# DPO_v4 Targeted Preference Data Report

## Purpose

This dataset is built from evaluation error cases of SFT_v5 and GRPO models. For each wrong model output, the official GSM8K solution is used as chosen and the model's incorrect generation is used as rejected.

## Input Sources

- sft_v5_limit200: files=1, flexible_rows=200, wrong_rows=80, added=80
- grpo_v6_quick50_limit200: files=1, flexible_rows=200, wrong_rows=77, added=77
- grpo_v9_ckpt50_limit200: files=1, flexible_rows=200, wrong_rows=78, added=78

## Output

- train_file: `data/processed/dpo_v4_targeted_from_sft_grpo_errors.jsonl`
- preview_file: `data/samples/dpo_v4_targeted_from_sft_grpo_errors_preview.jsonl`
- total_records: 235

## Counts by Source

- sft_v5_limit200: 80
- grpo_v6_quick50_limit200: 77
- grpo_v9_ckpt50_limit200: 78

## Fields

- prompt: eval-style question prompt
- chosen: official GSM8K gold solution
- rejected: incorrect model generation
- target: final numeric answer
- source: model/eval source
- doc_id: GSM8K sample id in lm-eval
