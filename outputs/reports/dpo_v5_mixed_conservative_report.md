# DPO_v5 Mixed Conservative Data Report

## Input Source Stats

- sft_v5_limit200: {'files': 1, 'flexible_rows': 200, 'correct_rows': 120, 'wrong_rows': 80}
- grpo_v6_quick50_limit200: {'files': 1, 'flexible_rows': 200, 'correct_rows': 123, 'wrong_rows': 77}
- grpo_v9_ckpt50_limit200: {'files': 1, 'flexible_rows': 200, 'correct_rows': 122, 'wrong_rows': 78}

## Pair Construction

- raw_hard_pairs: 235
- raw_safe_pairs: 35
- selected_max_hard: 120
- selected_max_safe: 80

## Final Counts

- total_records: 155
- hard_gold_vs_wrong: 120
- safe_correct_vs_wrong: 35

## Output

- train_file: `data/processed/dpo_v5_mixed_conservative.jsonl`
- preview_file: `data/samples/dpo_v5_mixed_conservative_preview.jsonl`