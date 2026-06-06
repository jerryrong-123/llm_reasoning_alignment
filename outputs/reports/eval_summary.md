# Evaluation Summary

> 注意：当前结果来自 debug 设置，例如 limit=5、max_steps=1，只能证明流程跑通，不能作为正式模型性能。

| Stage | Task | Metric | Value | Stderr |
|---|---|---|---:|---:|
| baseline | gsm8k_cot | exact_match,flexible-extract | 0.8000 | 0.2000 |
| baseline | gsm8k_cot | exact_match,flexible-extract | 0.8000 | 0.2000 |
| baseline | gsm8k_cot | exact_match,strict-match | 0.6000 | 0.2449 |
| baseline | gsm8k_cot | exact_match,strict-match | 0.6000 | 0.2449 |
| baseline | gsm8k_cot | sample_len | 5 | None |
| baseline | gsm8k_cot | sample_len | 5 | None |
| sft_lora | gsm8k_cot | exact_match,flexible-extract | 0.8000 | 0.2000 |
| sft_lora | gsm8k_cot | exact_match,flexible-extract | 0.8000 | 0.2000 |
| sft_lora | gsm8k_cot | exact_match,strict-match | 0.4000 | 0.2449 |
| sft_lora | gsm8k_cot | exact_match,strict-match | 0.4000 | 0.2449 |
| sft_lora | gsm8k_cot | sample_len | 5 | None |
| sft_lora | gsm8k_cot | sample_len | 5 | None |
| dpo_lora | gsm8k_cot | exact_match,flexible-extract | 0.8000 | 0.2000 |
| dpo_lora | gsm8k_cot | exact_match,strict-match | 0.4000 | 0.2449 |
| dpo_lora | gsm8k_cot | sample_len | 5 | None |
| grpo_lora | gsm8k_cot | exact_match,flexible-extract | 0.8000 | 0.2000 |
| grpo_lora | gsm8k_cot | exact_match,strict-match | 0.4000 | 0.2449 |
| grpo_lora | gsm8k_cot | sample_len | 5 | None |
| sft_lora_small | gsm8k_cot | exact_match,flexible-extract | 0.4500 | 0.1141 |
| grpo_lora_small | gsm8k_cot | exact_match,flexible-extract | 0.4000 | 0.1124 |
| dpo_lora_small | gsm8k_cot | exact_match,flexible-extract | 0.4000 | 0.1124 |
| sft_lora_small | gsm8k_cot | exact_match,strict-match | 0.2500 | 0.0993 |
| grpo_lora_small | gsm8k_cot | exact_match,strict-match | 0.2000 | 0.0918 |
| dpo_lora_small | gsm8k_cot | exact_match,strict-match | 0.2000 | 0.0918 |
| sft_lora_small | gsm8k_cot | sample_len | 20 | None |
| grpo_lora_small | gsm8k_cot | sample_len | 20 | None |
| dpo_lora_small | gsm8k_cot | sample_len | 20 | None |