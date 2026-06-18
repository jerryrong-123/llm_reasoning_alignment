# DPO_v6 Same-style Correct-vs-Wrong Report

## Source Stats
- sft_v5_limit200: {'files': 1, 'flexible_rows': 200, 'correct_rows': 120, 'wrong_rows': 80}
- grpo_v6_quick50_limit200: {'files': 1, 'flexible_rows': 200, 'correct_rows': 123, 'wrong_rows': 77}
- grpo_v9_ckpt50_limit200: {'files': 1, 'flexible_rows': 200, 'correct_rows': 122, 'wrong_rows': 78}

## Output
- train_file: `data/processed/dpo_v6_same_style_correct_vs_wrong.jsonl`
- preview_file: `data/samples/dpo_v6_same_style_correct_vs_wrong_preview.jsonl`
- total_records: 35

## Meaning

chosen is a model-generated correct response, rejected is a model-generated wrong response. This avoids forcing the model to imitate gold-answer style directly.