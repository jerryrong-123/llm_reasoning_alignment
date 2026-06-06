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