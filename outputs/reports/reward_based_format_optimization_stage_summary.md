# Reward-based format optimization 阶段总结报告

## 阶段目标

本阶段目标是解决前面 prompt-level format optimization 的问题。

之前的实验发现：

```text
1. 强制 prompt 输出格式可能提升格式命中率；
2. 但 prompt-level format optimization 会干扰推理正确率；
3. 单纯要求模型输出格式，不一定能稳定提升 strict / format 指标；
4. 因此更合理的方向是 reward-based format optimization。

所以本阶段引入 final-answer reward，并尝试把它接入 GRPO/RLVR 训练。

Reward 设计

本阶段使用的 reward 模块是：

src/rewards/final_answer_reward.py

核心 reward 结构是：

correctness_reward:
  +1.0 if extracted numeric answer equals gold answer

format_reward:
  +0.1 if response contains a clear final-answer line

extractability_reward:
  +0.1 if a numeric prediction can be extracted

total_reward = correctness_reward + format_reward + extractability_reward

核心原则是：

answer correctness reward > final-answer format reward

也就是说，答案正确性是主 reward，格式只是辅助 reward。

这样可以避免模型只学会输出 Final answer: ...，但实际答案仍然错误。

第一阶段：final-answer reward 测试

相关文件：

src/rewards/final_answer_reward.py
scripts/30_test_final_answer_reward.py
outputs/reports/final_answer_reward_test.md

测试结果：

correctness_rate = 0.5500
format_hit_rate = 0.3500
extractable_rate = 1.0000
avg_total_reward = 0.6850

结论：

final-answer reward 可以正常计算；
reward 中 correctness_reward 保持主导地位；
format_reward 不会压过答案正确性；
可以进入 GRPO 接入测试。
第二阶段：从 dpo_lora_small 出发的 GRPO format reward
Debug v1

相关文件：

configs/grpo_format_reward_debug.yaml
scripts/31_train_grpo_format_reward_debug.py
outputs/reports/grpo_format_reward_debug_report.md

关键结果：

reward_mean = 0.1
reward_std = 0
frac_reward_zero_std = 1
loss = 0
grad_norm = 0
adapter_config.json = True

结论：

final-answer reward 成功接入 GRPOTrainer；
adapter 可以保存；
但 reward_std = 0，没有有效学习信号。
Reward inspection

相关文件：

scripts/32_inspect_grpo_format_reward.py
outputs/reports/grpo_format_reward_inspection.md

v2 inspection 结果：

total_generations = 12
correctness_rate = 0.5833
format_hit_rate = 0.5000
extractable_rate = 1.0000
avg_total_reward = 0.7333
unique_reward_values = [0.1, 0.2, 1.1, 1.2]

结论：

更明确的 format prompt 和更长生成长度可以产生 reward 方差。
Debug v2

相关文件：

scripts/33_prepare_grpo_format_reward_data.py
configs/grpo_format_reward_debug.yaml
outputs/reports/grpo_format_reward_debug_v2_report.md

关键结果：

reward_mean = 0.6
reward_std = 0.5774
frac_reward_zero_std = 0
grad_norm = 0.7355
adapter_config.json = True

结论：

format-reward 专用数据是必要的；
num_generations=4 比 num_generations=2 更适合这个 debug；
max_completion_length=256 能产生有效 reward 方差。
Small run

相关文件：

configs/grpo_format_reward_small.yaml
outputs/reports/grpo_format_reward_small_report.md
outputs/checkpoints/grpo_lora_small_format_reward

5 step 训练结果：

step 1 reward_std = 0.5774, grad_norm = 0.7355
step 2 reward_std = 0.5354, grad_norm = 1.584
step 3 reward_std = 0,      grad_norm = 0
step 4 reward_std = 0.5,    grad_norm = 2.829
step 5 reward_std = 0.05,   grad_norm = 0.788

评估结果：

grpo_lora_small_format_reward:
flexible = 0.4000
strict   = 0.2000

对比：

sft_lora_small_v2:
flexible = 0.6000
strict   = 0.2000

结论：

从 dpo_lora_small 出发的 GRPO format reward 训练链路成功；
但最终 GSM8K-COT 指标没有超过 sft_lora_small_v2。
第三阶段：从 sft_lora_small_v2 出发的 GRPO format reward

因为当前最佳 checkpoint 是：

outputs/checkpoints/sft_lora_small_v2

所以继续测试：

sft_lora_small_v2 -> GRPO format reward
256 debug

相关文件：

configs/grpo_format_reward_from_sft_v2_debug.yaml
scripts/36_train_grpo_format_reward_from_adapter_debug.py
outputs/reports/grpo_format_reward_from_sft_v2_debug_report.md

关键结果：

reward_mean = 0.1
reward_std = 0
frac_reward_zero_std = 1
loss = 0
grad_norm = 0
adapter_config.json = True

结论：

从 sft_lora_small_v2 出发的训练链路可以跑通；
但 max_completion_length=256 时没有有效 reward 方差。
SFT-v2 reward inspection

相关文件：

scripts/37_inspect_sft_v2_format_reward.py
outputs/reports/sft_v2_format_reward_inspection.md

inspection 结果：

total_generations = 12
correctness_count = 6
correctness_rate = 0.5000
format_hit_count = 5
format_hit_rate = 0.4167
extractable_count = 12
extractable_rate = 1.0000
avg_total_reward = 0.6417
unique_reward_values = [0.1, 0.2, 1.1, 1.2]

结论：

sft_lora_small_v2 不是不能产生 reward 方差；
问题主要在生成长度和截断；
当 max_new_tokens=384 时，可以观察到不同 reward。
384 debug

相关文件：

configs/grpo_format_reward_from_sft_v2_debug_384.yaml
outputs/reports/grpo_format_reward_from_sft_v2_debug_384_report.md

关键结果：

reward_std = 0.5774
frac_reward_zero_std = 0
grad_norm = 0.7084
adapter_config.json = True

结论：

max_completion_length=384 可以让从 sft_lora_small_v2 出发的 GRPO debug 产生有效 reward 方差。
384 small run

相关文件：

configs/grpo_format_reward_from_sft_v2_384_small.yaml
outputs/reports/grpo_format_reward_from_sft_v2_384_small_report.md
outputs/checkpoints/grpo_lora_small_v2_format_reward_384

5 step 训练结果：

step 1 reward_mean = 0.600, reward_std = 0.5774, grad_norm = 0.7084
step 2 reward_mean = 0.650, reward_std = 0.5802, grad_norm = 0.7045
step 3 reward_mean = 0.100, reward_std = 0,      grad_norm = 0
step 4 reward_mean = 0.375, reward_std = 0.55,   grad_norm = 0.6637
step 5 reward_mean = 0.125, reward_std = 0.05,   grad_norm = 0.9398

结论：

从 sft_lora_small_v2 出发；
使用 max_completion_length=384；
进行 5 step GRPO format reward；
训练链路成功，并且多数 step 有有效 reward 方差。
第四阶段：评估 grpo_lora_small_v2_format_reward_384

相关文件：

configs/eval_grpo_sft_v2_format_reward_384_lora.yaml
outputs/reports/grpo_sft_v2_format_reward_384_eval_report.md

评估结果：

grpo_lora_small_v2_format_reward_384:
flexible = 0.6000
strict   = 0.2000

对比：

sft_lora_small_v2:
flexible = 0.6000
strict   = 0.2000

grpo_lora_small_format_reward:
flexible = 0.4000
strict   = 0.2000

grpo_lora_small_v2_format_reward_384:
flexible = 0.6000
strict   = 0.2000

结论：

从 sft_lora_small_v2 出发明显优于从 dpo_lora_small 出发；
max_completion_length=384 可以恢复最终评估表现；
但当前 5-step GRPO 仍然只是追平 sft_lora_small_v2，没有超过。
第五阶段：SFT-v2 vs GRPO-384 样本级对比

相关文件：

scripts/38_compare_sft_v2_vs_grpo_384_samples.py
outputs/reports/sft_v2_vs_grpo_384_sample_comparison.md

精确读取的 sample 文件：

sft_lora_small_v2:
outputs/eval/sft_lora_small_v2_qwen25_15b_gsm8k_cot_limit20/outputs__checkpoints__sft_lora_small_v2/samples_gsm8k_cot_2026-06-07T19-15-00.209160.jsonl

grpo_lora_small_v2_format_reward_384:
outputs/eval/grpo_lora_small_v2_format_reward_384_qwen25_15b_gsm8k_cot_limit20/outputs__checkpoints__grpo_lora_small_v2_format_reward_384/samples_gsm8k_cot_2026-06-08T21-44-10.128064.jsonl

样本级结果：

sft_lora_small_v2: 12/20 = 0.6000
grpo_lora_small_v2_format_reward_384: 12/20 = 0.6000

分类结果：

both_correct = 12
both_wrong = 8
grpo_improved = 0
grpo_regressed = 0

结论：

GRPO-384 和 SFT small_v2 不只是总分一样；
它们在当前 20 个 GSM8K-COT 样本上逐题表现完全一致。
当前阶段总论

本阶段最终结论是：

reward-based format optimization 链路已经跑通，
但当前 small-scale GRPO format-reward 训练还没有超过最佳 SFT checkpoint。

更细的结论是：

1. final-answer reward 可以正常计算；
2. final-answer reward 可以接入 GRPOTrainer；
3. format-reward 专用 prompt 数据是必要的；
4. num_generations=4 可以产生组内比较；
5. max_completion_length=256 对 sft_lora_small_v2 不够；
6. max_completion_length=384 可以恢复 reward 方差；
7. 从 sft_lora_small_v2 出发优于从 dpo_lora_small 出发；
8. 当前 GRPO-384 最终指标追平 sft_lora_small_v2；
9. 逐题对比显示 GRPO-384 没有新增正确样本；
10. 下一步不应该继续盲目加 GRPO step。