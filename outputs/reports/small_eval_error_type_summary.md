# Small Evaluation Error Type Summary

> 本报告基于 small_eval_error_analysis.csv，用于区分 small 阶段错误到底是格式问题，还是数学推理 / 计算问题。

## Summary

| Stage | Error Type | Count | Ratio |
|---|---|---:|---:|
| sft_lora_small | correct | 5 | 0.2500 |
| sft_lora_small | format_only_error | 4 | 0.2000 |
| sft_lora_small | reasoning_or_calc_error | 11 | 0.5500 |
| sft_lora_small | unknown | 0 | 0.0000 |
| dpo_lora_small | correct | 3 | 0.1500 |
| dpo_lora_small | format_only_error | 5 | 0.2500 |
| dpo_lora_small | reasoning_or_calc_error | 12 | 0.6000 |
| dpo_lora_small | unknown | 0 | 0.0000 |
| grpo_lora_small | correct | 3 | 0.1500 |
| grpo_lora_small | format_only_error | 5 | 0.2500 |
| grpo_lora_small | reasoning_or_calc_error | 12 | 0.6000 |
| grpo_lora_small | unknown | 0 | 0.0000 |

## Interpretation

- `format_only_error`：答案数值已经对了，但 strict-match 不通过，说明需要优化输出格式。
- `reasoning_or_calc_error`：答案数值本身错了，说明是题意理解、推理链或计算错误。
- 如果 format_only_error 较多，下一步应优先做格式约束实验。
- 如果 reasoning_or_calc_error 较多，下一步应优先改进训练数据、DPO 数据质量、reward 或训练步数。

