# SFT small_v2 vs GRPO 384 sample comparison

## 实验目的

本报告逐题对比 `sft_lora_small_v2` 和 `grpo_lora_small_v2_format_reward_384` 的 lm-eval sample 输出。

二者官方 lm-eval 指标相同：

```text
flexible-extract = 0.6000
strict-match     = 0.2000
```

本报告用于判断它们是逐题完全一致，还是只是总分一致。

## 精确 sample 文件核对

| model | sample_file |
|---|---|
| sft_lora_small_v2 | outputs/eval/sft_lora_small_v2_qwen25_15b_gsm8k_cot_limit20/outputs__checkpoints__sft_lora_small_v2/samples_gsm8k_cot_2026-06-07T19-15-00.209160.jsonl |
| grpo_lora_small_v2_format_reward_384 | outputs/eval/grpo_lora_small_v2_format_reward_384_qwen25_15b_gsm8k_cot_limit20/outputs__checkpoints__grpo_lora_small_v2_format_reward_384/samples_gsm8k_cot_2026-06-08T21-44-10.128064.jsonl |

## 准确率汇总

| model | correct | total | acc |
|---|---:|---:|---:|
| sft_lora_small_v2 | 12 | 20 | 0.6000 |
| grpo_lora_small_v2_format_reward_384 | 12 | 20 | 0.6000 |

## 分类汇总

| category | count |
|---|---:|
| both_correct | 12 |
| both_wrong | 8 |

## 结论解释

- `both_correct` 表示两个模型都答对。
- `both_wrong` 表示两个模型都答错。
- `grpo_improved` 表示 SFT-v2 错，但 GRPO-384 对。
- `grpo_regressed` 表示 SFT-v2 对，但 GRPO-384 错。

如果 `grpo_improved` 和 `grpo_regressed` 都为 0，说明两者逐题表现完全一致。
如果二者都大于 0，说明总分相同但样本分布有变化。

## Case table

| doc_id | gold | sft_pred | sft_ok | grpo_384_pred | grpo_384_ok | category |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 18 | 64 | 0 | 64 | 0 | both_wrong |
| 1 | 3 | 3 | 1 | 3 | 1 | both_correct |
| 2 | 70000 | 70000 | 1 | 70000 | 1 | both_correct |
| 3 | 540 | 540 | 1 | 540 | 1 | both_correct |
| 4 | 20 | 20 | 1 | 20 | 1 | both_correct |
| 5 | 64 | 29 | 0 | 29 | 0 | both_wrong |
| 6 | 260 | 260 | 1 | 260 | 1 | both_correct |
| 7 | 160 | 296 | 0 | 296 | 0 | both_wrong |
| 8 | 45 | 2 | 0 | 2 | 0 | both_wrong |
| 9 | 460 | 460 | 1 | 460 | 1 | both_correct |
| 10 | 366 | 366 | 1 | 366 | 1 | both_correct |
| 11 | 694 | 1588 | 0 | 1588 | 0 | both_wrong |
| 12 | 13 | 63 | 0 | 033333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333 | 0 | both_wrong |
| 13 | 18 | 36 | 0 | 36 | 0 | both_wrong |
| 14 | 60 | 60 | 1 | 60 | 1 | both_correct |
| 15 | 125 | 2971 | 0 | 2971 | 0 | both_wrong |
| 16 | 230 | 230 | 1 | 230 | 1 | both_correct |
| 17 | 57500 | 57500 | 1 | 57500 | 1 | both_correct |
| 18 | 7 | 7 | 1 | 7 | 1 | both_correct |
| 19 | 6 | 6 | 1 | 6 | 1 | both_correct |
