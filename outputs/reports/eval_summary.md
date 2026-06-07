# Evaluation Summary

> 注意：当前结果来自 debug / small 设置，例如 limit=5/20、max_steps=1/10/20。这些结果用于验证 SFT-DPO-GRPO/RLVR 闭环流程和小规模对比，不能作为正式模型性能。

| Stage | Task | Sample Len | Metric | Value | Stderr |
|---|---|---:|---|---:|---:|
| baseline | gsm8k_cot | 5 | exact_match,flexible-extract | 0.8000 | 0.2000 |
| baseline | gsm8k_cot | 5 | exact_match,strict-match | 0.6000 | 0.2449 |
| baseline | gsm8k_cot | 5 | sample_len | 5 | None |
| sft_lora | gsm8k_cot | 5 | exact_match,flexible-extract | 0.8000 | 0.2000 |
| sft_lora | gsm8k_cot | 5 | exact_match,strict-match | 0.4000 | 0.2449 |
| sft_lora | gsm8k_cot | 5 | sample_len | 5 | None |
| dpo_lora | gsm8k_cot | 5 | exact_match,flexible-extract | 0.8000 | 0.2000 |
| dpo_lora | gsm8k_cot | 5 | exact_match,strict-match | 0.4000 | 0.2449 |
| dpo_lora | gsm8k_cot | 5 | sample_len | 5 | None |
| grpo_lora | gsm8k_cot | 5 | exact_match,flexible-extract | 0.8000 | 0.2000 |
| grpo_lora | gsm8k_cot | 5 | exact_match,strict-match | 0.4000 | 0.2449 |
| grpo_lora | gsm8k_cot | 5 | sample_len | 5 | None |
| sft_lora_small | gsm8k_cot | 20 | exact_match,flexible-extract | 0.4500 | 0.1141 |
| sft_lora_small | gsm8k_cot | 20 | exact_match,strict-match | 0.2500 | 0.0993 |
| sft_lora_small | gsm8k_cot | 20 | sample_len | 20 | None |
| dpo_lora_small | gsm8k_cot | 20 | exact_match,flexible-extract | 0.4000 | 0.1124 |
| dpo_lora_small | gsm8k_cot | 20 | exact_match,strict-match | 0.2000 | 0.0918 |
| dpo_lora_small | gsm8k_cot | 20 | sample_len | 20 | None |
| grpo_lora_small | gsm8k_cot | 20 | exact_match,flexible-extract | 0.4000 | 0.1124 |
| grpo_lora_small | gsm8k_cot | 20 | exact_match,strict-match | 0.2000 | 0.0918 |
| grpo_lora_small | gsm8k_cot | 20 | sample_len | 20 | None |
| sft_lora_small_v2 | gsm8k_cot | 20 | exact_match,flexible-extract | 0.6000 | 0.1124 |
| sft_lora_small_v2 | gsm8k_cot | 20 | exact_match,strict-match | 0.2000 | 0.0918 |
| sft_lora_small_v2 | gsm8k_cot | 20 | sample_len | 20 | None |
