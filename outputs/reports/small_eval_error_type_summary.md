# Small Evaluation Error Type Summary

> 本报告基于 small_eval_error_analysis.csv，用于区分 small 阶段错误到底是格式问题、答案抽取问题，还是数学推理 / 计算问题。

## Summary

| Stage | Error Type | Count | Ratio |
|---|---|---:|---:|
| sft_lora_small | correct | 5 | 0.2500 |
| sft_lora_small | strict_format_only_error | 4 | 0.2000 |
| sft_lora_small | answer_extraction_or_format_error | 0 | 0.0000 |
| sft_lora_small | reasoning_or_calc_error | 11 | 0.5500 |
| sft_lora_small | unknown | 0 | 0.0000 |
| dpo_lora_small | correct | 3 | 0.1500 |
| dpo_lora_small | strict_format_only_error | 5 | 0.2500 |
| dpo_lora_small | answer_extraction_or_format_error | 1 | 0.0500 |
| dpo_lora_small | reasoning_or_calc_error | 11 | 0.5500 |
| dpo_lora_small | unknown | 0 | 0.0000 |
| grpo_lora_small | correct | 3 | 0.1500 |
| grpo_lora_small | strict_format_only_error | 5 | 0.2500 |
| grpo_lora_small | answer_extraction_or_format_error | 1 | 0.0500 |
| grpo_lora_small | reasoning_or_calc_error | 11 | 0.5500 |
| grpo_lora_small | unknown | 0 | 0.0000 |

## Interpretation

- `correct`：flexible-extract 和 strict-match 都通过。
- `strict_format_only_error`：答案数值正确，flexible-extract 通过，但 strict-match 不通过，主要是 strict 输出格式问题。
- `answer_extraction_or_format_error`：lm-eval 的 flexible-extract 判错，但脚本抽取的 pred_answer 与 gold_answer 实际相等，可能是答案抽取或格式兼容问题。
- `reasoning_or_calc_error`：pred_answer 与 gold_answer 不一致，说明答案数值本身错误，更可能是题意理解、推理链或计算错误。
- `unknown`：无法根据当前字段判断。

## Next Step

后续 reasoning 错误模式分析应该只针对 `reasoning_or_calc_error`，不要把 `answer_extraction_or_format_error` 误算成真正推理错误。

