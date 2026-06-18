# SFT small_v2 prompt-level format eval v2

## 实验目的

本实验不重新训练模型，只评估当前最佳 checkpoint `outputs/checkpoints/sft_lora_small_v2`。

相比 v1 的强制 `#### <answer>`，v2 只在 prompt 末尾温和要求输出 `Final answer: <answer>`，用于测试是否能减少格式约束对推理正确率的干扰。

## Summary

| metric | value |
|---|---:|
| total | 20 |
| flexible_correct | 11 |
| flexible_acc | 0.5500 |
| final_answer_format_hit | 14 |
| final_answer_format_hit_rate | 0.7000 |
| final_answer_format_correct | 8 |
| format_acc | 0.4000 |
| strict_hash_correct | 0 |
| strict_hash_acc | 0.0000 |

## Interpretation

- 如果 `flexible_acc >= 0.55` 且 `format_acc >= 0.30`，说明温和 prompt 有继续优化价值。
- 如果 `flexible_acc` 仍明显低于原始 `sft_lora_small_v2` 的 `0.6000`，说明 prompt-level format optimization 仍然会伤害推理，应转向 reward-based format optimization。

## Cases

| doc_id | gold | flexible_pred | final_answer_pred | flexible_correct | final_answer_format_correct |
|---:|---:|---:|---:|---:|---:|
| 0 | 18 | -10 | None | 0 | 0 |
| 1 | 3 | 3 | 3 | 1 | 1 |
| 2 | 70000 | 70,000 | None | 1 | 0 |
| 3 | 540 | 540 | 540 | 1 | 1 |
| 4 | 20 | 20 | 20 | 1 | 1 |
| 5 | 64 | 44 | 44 | 0 | 0 |
| 6 | 260 | 260 | 260 | 1 | 1 |
| 7 | 160 | 120 | 120 | 0 | 0 |
| 8 | 45 | -160 | -160 | 0 | 0 |
| 9 | 460 | 460 | None | 1 | 0 |
| 10 | 366 | 294 | 294 | 0 | 0 |
| 11 | 694 | 2233 | 2233 | 0 | 0 |
| 12 | 13 | 10 | 10 | 0 | 0 |
| 13 | 18 | 10 | None | 0 | 0 |
| 14 | 60 | 60 | 60 | 1 | 1 |
| 15 | 125 | 125 | None | 1 | 0 |
| 16 | 230 | 230 | 230 | 1 | 1 |
| 17 | 57500 | 57500 | 57500 | 1 | 1 |
| 18 | 7 | 7 | 7 | 1 | 1 |
| 19 | 6 | 12 | None | 0 | 0 |